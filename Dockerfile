# Dockerfile

# Step 1: Start with a stable, official Python base image.
FROM python:3.10-slim

# Step 2: Install system-level dependencies.
# PyMuPDF often needs these libraries on Debian/Ubuntu-based systems.
# This prevents potential installation errors for the 'pymupdf' package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory inside the container.
WORKDIR /app

# Step 4: Copy the requirements file into the container.
# This is done first to leverage Docker's layer caching.
COPY requirements.txt .

# Step 5: Install the Python dependencies from the requirements file.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code (app.py) into the container.
COPY . .

# Step 7: Expose the default port that Streamlit runs on.
EXPOSE 8501

# Step 8: The command to run when the container starts.
# This starts the Streamlit server and makes it accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]