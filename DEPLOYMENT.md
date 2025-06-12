# Open WebUI CI/CD Deployment Guide

This guide explains how to set up automated deployment of Open WebUI from GitHub Actions to a DigitalOcean Droplet.

## Architecture Overview

```
GitHub Repository → GitHub Actions → GitHub Container Registry → DigitalOcean Droplet
```

The CI/CD pipeline:
1. Builds a Docker image when code is pushed to the `main` branch
2. Pushes the image to GitHub Container Registry
3. Deploys the image to your DigitalOcean Droplet via SSH
4. Runs health checks to verify deployment

## Prerequisites

- DigitalOcean Droplet with Ubuntu 22.04+
- SSH access to the Droplet
- GitHub repository with this Open WebUI code

## Setup Instructions

### 1. Prepare Your Droplet

Run the setup script on your Droplet to install all dependencies:

```bash
# SSH into your droplet
ssh root@143.198.117.242

# Download and run the setup script
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/.github/scripts/setup-droplet.sh | bash

# Or run it manually if you prefer:
wget https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/.github/scripts/setup-droplet.sh
chmod +x setup-droplet.sh
./setup-droplet.sh
```

### 2. Configure GitHub Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DROPLET_HOST` | `143.198.117.242` | Your Droplet's IP address |
| `DROPLET_USERNAME` | `root` | SSH username (usually root) |
| `DROPLET_SSH_KEY` | `-----BEGIN OPENSSH PRIVATE KEY-----...` | Your private SSH key |
| `WEBUI_SECRET_KEY` | `your-random-secret-key` | Secret key for Open WebUI (generate a random string) |

#### Generating SSH Key (if needed)

If you don't have an SSH key pair:

```bash
# Generate a new SSH key pair
ssh-keygen -t ed25519 -C "your-email@example.com" -f ~/.ssh/droplet_key

# Copy the public key to your droplet
ssh-copy-id -i ~/.ssh/droplet_key.pub root@143.198.117.242

# Use the private key content for DROPLET_SSH_KEY secret
cat ~/.ssh/droplet_key
```

#### Generating Secret Key

```bash
# Generate a random secret key
openssl rand -hex 32
```

### 3. Deploy

Once everything is configured:

1. Push your code to the `main` branch
2. GitHub Actions will automatically build and deploy
3. Monitor the deployment in the Actions tab
4. Access your Open WebUI at: http://143.198.117.242

## Manual Deployment

If you need to deploy manually:

```bash
# SSH into your droplet
ssh root@143.198.117.242

# Navigate to the app directory
cd /opt/open-webui

# Pull latest changes and restart
docker-compose pull
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

## Monitoring and Maintenance

### Check Service Status

```bash
# Check if containers are running
docker-compose ps

# View logs
docker-compose logs -f open-webui
docker-compose logs -f ollama

# Check system resources
htop
df -h
```

### Backup Data

```bash
# Backup Open WebUI data
docker run --rm -v open-webui_open-webui-data:/data -v $(pwd):/backup ubuntu tar czf /backup/open-webui-backup-$(date +%Y%m%d).tar.gz -C /data .

# Backup Ollama models
docker run --rm -v open-webui_ollama-data:/data -v $(pwd):/backup ubuntu tar czf /backup/ollama-backup-$(date +%Y%m%d).tar.gz -C /data .
```

### Restore Data

```bash
# Restore Open WebUI data
docker run --rm -v open-webui_open-webui-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/open-webui-backup-YYYYMMDD.tar.gz -C /data

# Restore Ollama models
docker run --rm -v open-webui_ollama-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/ollama-backup-YYYYMMDD.tar.gz -C /data
```

## Troubleshooting

### Common Issues

1. **Deployment fails with SSH connection error**
   - Verify DROPLET_HOST, DROPLET_USERNAME, and DROPLET_SSH_KEY secrets
   - Ensure SSH key has proper permissions and is in OpenSSH format

2. **Container fails to start**
   - Check logs: `docker-compose logs`
   - Verify environment variables
   - Ensure sufficient disk space and memory

3. **Cannot access Open WebUI**
   - Check if containers are running: `docker-compose ps`
   - Verify firewall settings: `ufw status`
   - Check if port 80 is accessible

4. **Build fails**
   - Check GitHub Actions logs
   - Verify Dockerfile syntax
   - Ensure all dependencies are available

### Useful Commands

```bash
# Restart services
docker-compose restart

# Update to latest images
docker-compose pull && docker-compose up -d

# Clean up old images
docker image prune -f

# View resource usage
docker stats

# Access container shell
docker exec -it open-webui bash
docker exec -it ollama bash
```

## Security Considerations

- SSH key authentication (no passwords)
- Firewall configured to only allow necessary ports
- Automatic security updates enabled
- Docker containers run with non-root users where possible
- Regular backups recommended

## Scaling and Performance

For production use, consider:

- Using a load balancer for multiple Droplets
- Setting up monitoring with Prometheus/Grafana
- Implementing log aggregation
- Using a managed database for persistence
- Setting up SSL/TLS with Let's Encrypt

## Support

If you encounter issues:

1. Check the GitHub Actions logs
2. Review container logs on the Droplet
3. Verify all secrets are correctly configured
4. Ensure the Droplet has sufficient resources

For Open WebUI specific issues, refer to the [Open WebUI Documentation](https://docs.openwebui.com/).
