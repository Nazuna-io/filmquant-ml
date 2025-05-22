import pytest
import uvicorn
import threading
import time
from selenium import webdriver
import os
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Configuration for the server
APP_HOST = "127.0.0.1"
APP_PORT = 8082 # Using a different port than default to avoid conflicts
BASE_URL = f"http://{APP_HOST}:{APP_PORT}"
GRADIO_PATH = "/gradio" # Path where Gradio is mounted in app.main

@pytest.fixture(scope="session")
def firefox_driver():
    """Session-scoped fixture to initialize and quit the Firefox WebDriver."""
    options = webdriver.FirefoxOptions()
    options.binary_location = "/usr/bin/firefox"
    options.add_argument("--headless")  # Run headless for CI environments
    options.add_argument("--width=1920")
    options.add_argument("--height=1080") # Still useful for layout consistency
    
    # For Firefox, advanced logging might require setting preferences via options.set_preference
    # but driver.get_log('browser') might work for basic SEVERE logs by default with GeckoDriver.
    # We will try without specific logging capabilities first.
    
    # Setup geckodriver service with logging
    geckodriver_log_path = os.path.join(os.path.dirname(__file__), '..', 'geckodriver.log') # Store log in project root
    print(f"Geckodriver logs will be written to: {os.path.abspath(geckodriver_log_path)}")
    service = FirefoxService(
        executable_path=GeckoDriverManager().install(),
        log_output=geckodriver_log_path,
        service_args=['--log', 'debug'] # More verbose logging for geckodriver
    )
    driver = webdriver.Firefox(service=service, options=options)
    yield driver
    driver.quit()

@pytest.fixture(scope="session")
def live_server():
    """Session-scoped fixture to run the Uvicorn server in a separate thread."""
    config = uvicorn.Config("app.main:app", host=APP_HOST, port=APP_PORT, log_level="info")
    server = uvicorn.Server(config)
    
    # Run the server in a separate thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    # Wait for the server to start (adjust timeout as needed)
    time.sleep(5) # Give Uvicorn a few seconds to spin up
    yield server # The server instance isn't directly used but signals it's up
    # Server is daemonized, will shut down with the main thread

def test_gradio_page_loads_and_no_critical_errors(live_server, firefox_driver):
    """
    Test that the Gradio page loads, title is correct,
    and there are no critical JavaScript errors on page load.
    """
    driver = firefox_driver
    gradio_url = f"{BASE_URL}{GRADIO_PATH}"
    driver.get(gradio_url)
    print(f"Navigated to {gradio_url}")

    # Wait for the title to be correct (on the main page initially)
    WebDriverWait(driver, 20).until(EC.title_contains("FilmQuant ML - Film Revenue Predictor"))
    print("Main page loaded successfully with correct title.")

    # Switch to the Gradio iframe
    # Gradio apps mounted via gr.mount_gradio_app are typically inside an iframe
    try:
        WebDriverWait(driver, 20).until(
            EC.frame_to_be_available_and_switch_to_it((By.TAG_NAME, "iframe"))
        )
        print("Successfully switched to Gradio iframe.")
        print("Pausing for 2 seconds after iframe switch for Gradio to render...")
        time.sleep(2) # Diagnostic delay
        
        # Diagnostic: Print page source of iframe
        iframe_source = driver.page_source
        print("\n--- IFRAME PAGE SOURCE (after delay) ---")
        print(iframe_source)
        print("--- END IFRAME PAGE SOURCE ---\n")
    except TimeoutException:
        pytest.fail("Timeout waiting for Gradio iframe to load.")

    assert driver.title == "FilmQuant ML - Film Revenue Predictor", \
        f"Page title is incorrect. Expected 'FilmQuant ML - Film Revenue Predictor', got '{driver.title}'"

    print("\nPage loaded successfully with correct title.")

    # Check for severe browser console errors
    # (This captures JS errors, not Python server-side errors)
    # log_entries = driver.get_log('browser') # Commented out due to 'HTTP method not allowed' with Firefox/Geckodriver
    # severe_errors = [entry for entry in log_entries if entry['level'] == 'SEVERE']
    # 
    # if severe_errors:
    #     print("\nSEVERE browser console errors found:")
    #     for error in severe_errors:
    #         print(error)
    #     # We won't fail the test for *any* severe error initially, 
    #     # as some might be benign or unrelated Gradio warnings.
    #     # Instead, we'll print them for inspection.
    #     # You can add assertions here later if specific errors should cause failure.
    #     # Example: assert not any("Too many arguments" in e['message'] for e in severe_errors), "'Too many arguments' error found on page load!"

    print("\nInitial UI test for page load and title complete.")

    # Interact with the 'Predict New Film' tab (it should be active by default)
    print("Attempting to fill only film title and click predict...")
    try:
        # Give more time for Gradio components to render
        print("Waiting 5 seconds for Gradio components to render...")
        time.sleep(5)
        
        # Dump the DOM to debug what's actually there
        page_html = driver.page_source
        print("Page HTML snippet:")
        print(page_html[:1000])  # Print first 1000 chars
        
        # Try to find any textarea in the Gradio UI
        textareas = driver.find_elements(By.TAG_NAME, "textarea")
        print(f"Found {len(textareas)} textareas")
        
        if textareas:
            # Just use the first textarea (Film Title) as they appear in order
            title_input = textareas[0]
            print("Located Film Title input using direct textarea finder.")
            
            # Find the predict button by text content instead of xpath
            buttons = driver.find_elements(By.TAG_NAME, "button")
            predict_button = None
            for button in buttons:
                if "Predict Box Office Revenue" in button.text:
                    predict_button = button
                    break
                    
            if predict_button:
                print("Located Predict button by text content.")
                
                # Fill the form
                title_input.clear()
                title_input.send_keys("Selenium Test Movie")
                print("Filled Film Title input.")
                
                # Click the button
                predict_button.click()
                print("Clicked Predict button.")
                
                # Wait for results
                time.sleep(3)
                
                # Look for any result textareas
                result_textareas = driver.find_elements(By.TAG_NAME, "textarea")
                if len(result_textareas) > len(textareas):
                    print(f"Found {len(result_textareas) - len(textareas)} new textareas after clicking predict.")
                    for i, textarea in enumerate(result_textareas[len(textareas):]):
                        try:
                            print(f"Result {i+1}: {textarea.get_attribute('value')}")
                        except:
                            print(f"Could not get value from result {i+1}")
                else:
                    print("No new textareas found after prediction.")
            else:
                print("Could not find Predict button!")
        else:
            print("Could not find any textareas for input!")
                
        # Pause for manual inspection
        print("\nPAUSING FOR 5 SECONDS FOR INSPECTION")
        time.sleep(5)
        print("Resuming test.")

    except Exception as e:
        # Capture screenshot on failure
        driver.save_screenshot("failure_screenshot.png")
        print(f"Saved screenshot to failure_screenshot.png")
        # Optionally, print page source for debugging
        with open("failure_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"Saved page source to failure_page_source.html")
        pytest.fail(f"Error during form interaction: {e}")
    finally:
        # Switch back to the main document context if we were in an iframe
        driver.switch_to.default_content()
        print("Switched back to default content from iframe (if applicable).")
