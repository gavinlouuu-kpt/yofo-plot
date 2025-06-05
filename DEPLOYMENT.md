# Production Deployment Guide

## Requirements
- Python 3.8+
- Gunicorn
- Nginx (recommended)

## Installation
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export FLASK_ENV=production
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Running with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 "web_plot:create_app()"
```

## Nginx Configuration
1. Install Nginx:
```bash
sudo apt install nginx
```

2. Create a new config file at `/etc/nginx/sites-available/csv_plotter`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /path/to/your/app/static;
    }
}
```

3. Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/csv_plotter /etc/nginx/sites-enabled
sudo systemctl restart nginx
```

## Running as a Service
1. Create a systemd service file at `/etc/systemd/system/csv_plotter.service`:
```ini
[Unit]
Description=CSV Plotter Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/your/app
Environment="PYTHONPATH=/path/to/your/app"
ExecStart=/usr/bin/gunicorn -w 4 -b 127.0.0.1:5000 "web_plot:create_app()"

[Install]
WantedBy=multi-user.target
```

2. Start and enable the service:
```bash
sudo systemctl start csv_plotter
sudo systemctl enable csv_plotter
```

## Security Considerations
1. Set up HTTPS using Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

2. Configure automatic renewal:
```bash
sudo certbot renew --dry-run
