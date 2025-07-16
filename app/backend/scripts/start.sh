#!/bin/sh

PID=$(lsof -t -i @172.20.23.31:8000)
if [ ! -z "$PID" ]; then
  echo "Process found on 172.20.23.31:8000, killing it..."
  kill -9 $PID
fi
# Start your application
echo "Starting the application...\n\n"

python3 manage.py makemigrations
python3 manage.py migrate

python3 manage.py runserver 0.0.0.0:8000