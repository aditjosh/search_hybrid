# Use the official Eclipse Temurin image as the base
FROM eclipse-temurin:21-jdk

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default 'python' command (optional)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Ensure the virtual environment is used for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the application
CMD ["python", "app.py"]
