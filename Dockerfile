# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /usr/src/app
COPY ./app /usr/src/app/app
COPY ./data /usr/src/app/data

# Make port 8080 available to the world outside this container
# Gradio typically runs on 7860, Flask on 5000. Cloud Run expects 8080 by default.
# The Flask app will need to be configured to run on 0.0.0.0:8080
EXPOSE 8080

# Define environment variable for port, Gunicorn will use this
ENV PORT 8080

# Run app.main.py when the container launches
# Using gunicorn for production, assuming app.main:app is the Flask app instance
# CMD ["python", "app/main.py"] # For development
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.main:app
