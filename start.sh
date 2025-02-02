#!/bin/bash
gunicorn --timeout 300 --workers 4 --bind 0.0.0.0:5000 speechapi:app