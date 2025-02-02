#!/bin/bash
gunicorn --timeout 300 --workers 1 --bind 0.0.0.0:5000 speechapi:app