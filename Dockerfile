FROM ultralytics/ultralytics:latest-jetson-jetpack6
# FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /app

RUN pip install "setuptools<69" && \
    pip install anomalib dotenv einops FrEIA kornia lightning open-clip-torch scikit-image tifffile timm && \
    pip install -U setuptools

# # Set the entry point
# CMD ["python", "main.py"]
