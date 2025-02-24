
# Use Python 3.12.7 as base image
FROM python:3.12.7

# Set the working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Ensure Python finds `src/`
ENV PYTHONPATH="/app/src"

# Run the script
CMD ["python", "main_docker.py"]