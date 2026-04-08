#!/bin/bash
trap 'kill $(jobs -p)' EXIT

python manage.py migrate --noinput
python manage.py qcluster &
gunicorn animal.wsgi --bind=0.0.0.0:${PORT:-8000} --timeout 600 -w 1 --access-logfile '-' --error-logfile '-'
