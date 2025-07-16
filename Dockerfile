FROM ultralytics/ultralytics:latest-jetson-jetpack6
# FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /app

# Set the entry point
CMD ["python", "main.py"]
