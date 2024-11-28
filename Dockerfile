# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10.0-slim

# Set the working directory in the container
WORKDIR /app

# Install pipenv, cmake, and build tools
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock ./

# Install the dependencies using pipenv
RUN pipenv install --deploy --ignore-pipfile --python /usr/local/bin/python

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=lab/backend/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port the app runs on
EXPOSE 5001

# Run the application
CMD ["pipenv", "run", "flask", "run", "--host=0.0.0.0", "--port=5001"]