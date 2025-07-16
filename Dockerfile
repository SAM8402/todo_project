FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# 1. Install Python & SQLite support (globally, as root)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip libsqlite3-0 libsqlite3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# 2. Install your Python deps
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 3. Copy start.sh script into the image
COPY app/backend/scripts/start.sh /app/backend/scripts/start.sh

# 4. Give execute permissions to the start.sh script
RUN chmod +x /app/backend/scripts/start.sh


# 4. Create an unprivileged user and chown the app folder
RUN groupadd -g 1001 appGroup

RUN useradd -r -m -u 1001 -g appGroup appUser


# 5. Switch to that user
USER appUser

# 6. Expose port & default entrypoint
EXPOSE 8000

# 8. Set the default command to execute the start.sh script
CMD ["/app/backend/scripts/start.sh"]
