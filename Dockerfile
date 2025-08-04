FROM ultralytics/ultralytics:latest-jetson-jetpack6
# FROM ultralytics/ultralytics:latest

# ユーザーを作成
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# グループとユーザーを作成
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the working directory
WORKDIR /app

# ディレクトリの所有者を変更
RUN chown -R $USERNAME:$USERNAME /app

# Install cron and required packages, set timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y cron tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo "Asia/Tokyo" > /etc/timezone

RUN pip install "setuptools<69" && \
    pip install anomalib dotenv einops FrEIA imagemagick kornia lightning onnxslim open-clip-torch scikit-image tifffile timm && \
    pip install -U setuptools

# # Copy cron configuration
# COPY docker-crontab.txt /etc/cron.d/padim-yolo-cron

# # Set proper permissions for cron job
# RUN chmod 0644 /etc/cron.d/padim-yolo-cron && \
#     crontab /etc/cron.d/padim-yolo-cron

# Create logs directory
RUN mkdir -p /app/logs

# Copy and set permissions for startup script
COPY docker-start.sh /app/docker-start.sh
RUN chmod 755 /app/docker-start.sh

# ユーザーを切り替え
USER $USERNAME

# Set the entry point to start cron and keep container running
CMD ["/app/docker-start.sh"]
