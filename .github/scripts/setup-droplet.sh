#!/bin/bash

# Setup script for DigitalOcean Droplet
# This script prepares the droplet for Open WebUI deployment

set -e

echo "🚀 Setting up DigitalOcean Droplet for Open WebUI deployment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update
apt-get upgrade -y

# Install required packages
echo "🔧 Installing required packages..."
apt-get install -y curl wget git ufw

# Configure firewall
echo "🔒 Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Create application directory
echo "📁 Creating application directory..."
mkdir -p /opt/open-webui
chown -R $USER:$USER /opt/open-webui

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    
    # Add current user to docker group
    usermod -aG docker $USER
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
else
    echo "✅ Docker is already installed"
fi

# Install Docker Compose if not already installed
if ! command -v docker-compose &> /dev/null; then
    echo "🐙 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
else
    echo "✅ Docker Compose is already installed"
fi

# Create systemd service for automatic startup
echo "⚙️ Creating systemd service..."
cat > /etc/systemd/system/open-webui.service << 'EOF'
[Unit]
Description=Open WebUI
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/open-webui
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
systemctl daemon-reload
systemctl enable open-webui.service

# Create log rotation for Docker
echo "📝 Setting up log rotation..."
cat > /etc/logrotate.d/docker << 'EOF'
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size=1M
    missingok
    delaycompress
    copytruncate
}
EOF

# Set up automatic security updates
echo "🔐 Setting up automatic security updates..."
apt-get install -y unattended-upgrades
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

echo "✅ Droplet setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - DROPLET_HOST: $(curl -s http://checkip.amazonaws.com)"
echo "   - DROPLET_USERNAME: root"
echo "   - DROPLET_SSH_KEY: (your private SSH key)"
echo "   - WEBUI_SECRET_KEY: (generate a random secret key)"
echo ""
echo "2. Push your code to the main branch to trigger deployment"
echo ""
echo "🌐 Your Open WebUI will be available at: http://$(curl -s http://checkip.amazonaws.com)"
