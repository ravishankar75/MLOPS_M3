# Use a slim Python image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., required for some packages like scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install Python dependencies (Flask, pandas, scikit-learn, joblib)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app code into the container
COPY . /app/

# Copy the model files from the 'models' folder
COPY models/ /app/models/

# Expose the port that Flask will run on
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python", "app.py"]
