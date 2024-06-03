# Base Image: Start with a lightweight Python image
FROM python:3.11-slim

# Working Directory: Set up a workspace for your application
WORKDIR /app

# Copy Dependency File: Transfer only requirements.txt first to optimize Docker's caching
COPY requirements.txt requirements.txt

# Install Dependencies: Use pip to install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy App Code: Transfer the rest of your application code into the image
COPY . .

# Expose Port: Make your Flask app accessible on port 5000 (default for Flask)
EXPOSE 5000

# Run Command: Launch your Flask app using Gunicorn (production-ready WSGI server)
CMD ["python3", "app.py"]
