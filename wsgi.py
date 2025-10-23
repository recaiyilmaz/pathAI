"""
WSGI entry point for the SAM2 Flask application.
This file is used by deployment platforms like Render, Heroku, etc.
"""

from codeR import app

if __name__ == "__main__":
    app.run()
