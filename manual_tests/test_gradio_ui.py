import os
import sys
import time
import webbrowser

# Add the project root to the Python path to allow imports from 'app'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("Starting a simple manual test for FilmQuant ML Gradio UI...")
print("This will open a web browser to the Gradio UI so you can manually test it.")

# Assuming the server is already running on port 8081
url = "http://localhost:8081/gradio/"
print(f"Opening browser to {url}")
webbrowser.open(url)

print("\nInstructions:")
print(
    "1. In the Gradio UI, you should see the 'FilmQuant ML - Film Revenue Predictor' interface."
)
print("2. In the 'Film Title' field, enter 'Manual Test Movie'")
print("3. Leave all other fields at their default values (or empty)")
print("4. Click the 'Predict Box Office Revenue' button")
print("5. Verify that you get a prediction with no errors")
print("\nPress Ctrl+C when done testing")

# Keep the script running until Ctrl+C
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nTest completed. Exiting...")
