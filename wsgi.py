#!/usr/bin/env python3
"""
WSGI entry point for the SAM2 Image Segmentation application.
This file is used by gunicorn to serve the Flask application.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

print("WSGI: Starting to import codeR module...")

try:
    # Import the Flask app from codeR
    from codeR import app
    print("WSGI: Successfully imported Flask app from codeR")
except ImportError as e:
    print(f"WSGI: Error importing codeR: {e}")
    raise

# The Flask app instance that gunicorn will use
application = app

if __name__ == "__main__":
    # This allows running the WSGI file directly for testing
    port = int(os.environ.get('PORT', 5000))
    print(f"WSGI: Running Flask app directly on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)