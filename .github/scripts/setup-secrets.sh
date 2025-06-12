#!/bin/bash

# GitHub Secrets Setup Helper Script
# This script helps you set up the required GitHub secrets for deployment

set -e

echo "🔐 GitHub Secrets Setup Helper"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed.${NC}"
    echo "Please install it from: https://cli.github.com/"
    echo "Or use the GitHub web interface to add secrets manually."
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}⚠️  You need to authenticate with GitHub CLI first.${NC}"
    echo "Run: gh auth login"
    exit 1
fi

echo -e "${GREEN}✅ GitHub CLI is installed and authenticated${NC}"
echo ""

# Get repository information
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo -e "${BLUE}📁 Repository: ${REPO}${NC}"
echo ""

# Function to set a secret
set_secret() {
    local secret_name=$1
    local secret_description=$2
    local secret_value=$3
    
    if [ -z "$secret_value" ]; then
        echo -e "${YELLOW}⏭️  Skipping ${secret_name} (empty value)${NC}"
        return
    fi
    
    echo -e "${BLUE}Setting ${secret_name}...${NC}"
    if echo "$secret_value" | gh secret set "$secret_name"; then
        echo -e "${GREEN}✅ ${secret_name} set successfully${NC}"
    else
        echo -e "${RED}❌ Failed to set ${secret_name}${NC}"
    fi
}

# Collect secrets
echo "Please provide the following information:"
echo ""

# DROPLET_HOST
echo -e "${BLUE}1. Droplet IP Address${NC}"
read -p "Enter your DigitalOcean Droplet IP (e.g., 143.198.117.242): " DROPLET_HOST

# DROPLET_USERNAME
echo ""
echo -e "${BLUE}2. SSH Username${NC}"
read -p "Enter SSH username [root]: " DROPLET_USERNAME
DROPLET_USERNAME=${DROPLET_USERNAME:-root}

# DROPLET_SSH_KEY
echo ""
echo -e "${BLUE}3. SSH Private Key${NC}"
echo "Enter the path to your SSH private key file:"
read -p "SSH key path [~/.ssh/id_rsa]: " SSH_KEY_PATH
SSH_KEY_PATH=${SSH_KEY_PATH:-~/.ssh/id_rsa}

# Expand tilde to home directory
SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

if [ -f "$SSH_KEY_PATH" ]; then
    DROPLET_SSH_KEY=$(cat "$SSH_KEY_PATH")
    echo -e "${GREEN}✅ SSH key loaded from ${SSH_KEY_PATH}${NC}"
else
    echo -e "${RED}❌ SSH key file not found: ${SSH_KEY_PATH}${NC}"
    echo "Please enter your SSH private key manually:"
    echo "(Paste the entire key including -----BEGIN and -----END lines)"
    echo "Press Ctrl+D when done:"
    DROPLET_SSH_KEY=$(cat)
fi

# WEBUI_SECRET_KEY
echo ""
echo -e "${BLUE}4. WebUI Secret Key${NC}"
echo "Generating a random secret key..."
WEBUI_SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | base64)
echo -e "${GREEN}✅ Generated secret key: ${WEBUI_SECRET_KEY:0:16}...${NC}"

# Confirm before setting secrets
echo ""
echo -e "${YELLOW}📋 Summary of secrets to be set:${NC}"
echo "- DROPLET_HOST: $DROPLET_HOST"
echo "- DROPLET_USERNAME: $DROPLET_USERNAME"
echo "- DROPLET_SSH_KEY: [SSH private key]"
echo "- WEBUI_SECRET_KEY: ${WEBUI_SECRET_KEY:0:16}..."
echo ""

read -p "Do you want to set these secrets? (y/N): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

# Set the secrets
echo ""
echo -e "${BLUE}🔐 Setting GitHub secrets...${NC}"
echo ""

set_secret "DROPLET_HOST" "DigitalOcean Droplet IP address" "$DROPLET_HOST"
set_secret "DROPLET_USERNAME" "SSH username for the droplet" "$DROPLET_USERNAME"
set_secret "DROPLET_SSH_KEY" "SSH private key for droplet access" "$DROPLET_SSH_KEY"
set_secret "WEBUI_SECRET_KEY" "Secret key for Open WebUI" "$WEBUI_SECRET_KEY"

echo ""
echo -e "${GREEN}🎉 All secrets have been set up!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Run the droplet setup script on your server:"
echo "   ssh $DROPLET_USERNAME@$DROPLET_HOST"
echo "   curl -fsSL https://raw.githubusercontent.com/$REPO/main/.github/scripts/setup-droplet.sh | bash"
echo ""
echo "2. Push your code to the main branch to trigger deployment:"
echo "   git add ."
echo "   git commit -m 'Add CI/CD pipeline'"
echo "   git push origin main"
echo ""
echo "3. Monitor the deployment in GitHub Actions:"
echo "   https://github.com/$REPO/actions"
echo ""
echo "4. Access your Open WebUI at:"
echo "   http://$DROPLET_HOST"
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
