#!/usr/bin/env python3
"""
Local development server runner.
For development only - production uses Gunicorn via Docker.
"""

import os
from web_plot import create_app

if __name__ == '__main__':
    # Ensure upload directory exists
    upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    
    # Create Flask app
    app = create_app()
    
    # Run with Flask development server
    debug_mode = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))
    
    print(f"Starting development server on {host}:{port}")
    print(f"Debug mode: {debug_mode}")
    app.run(host=host, port=port, debug=debug_mode) 