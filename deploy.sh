#!/bin/bash

# ============================================
#  VeriCheck - One-Click VPS Deployment Script
# ============================================

set -e

echo ""
echo "=========================================="
echo "  VeriCheck - VPS Deployment Script"
echo "=========================================="
echo ""
echo "This script will deploy VeriCheck on your VPS."
echo "Make sure you have SSH access to your server."
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "⚠ rsync not found. Using scp instead."
fi

# Get server details
read -p "Enter VPS IP address (e.g., 38.242.251.202): " VPS_IP
read -p "Enter VPS username (e.g., root): " VPS_USER
read -p "Enter deployment path on VPS (e.g., /var/www/vericheck): " VPS_PATH

echo ""
echo "=========================================="
echo "  Deployment Configuration"
echo "=========================================="
echo "  VPS IP:     $VPS_IP"
echo "  Username:   $VPS_USER"
echo "  Path:       $VPS_PATH"
echo "=========================================="
echo ""

read -p "Continue with deployment? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "📤 Uploading files to VPS..."

# Create remote directory
ssh "$VPS_USER@$VPS_IP" "mkdir -p $VPS_PATH"

# Upload files using rsync or scp
if command -v rsync &> /dev/null; then
    rsync -avz --exclude='venv' --exclude='node_modules' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' . "$VPS_USER@$VPS_IP:$VPS_PATH/"
else
    scp -r backend frontend data.csv memory.md plagiarism-checker-nlp.ipynb train_snli.txt README.md docker-compose.yml Dockerfile.backend nginx.conf setup.sh start.sh "$VPS_USER@$VPS_IP:$VPS_PATH/"
fi

echo ""
echo "🔧 Setting up on VPS..."

# SSH commands to set up Docker and deploy
ssh "$VPS_USER@$VPS_IP" << 'ENDSSH'
cd '"$VPS_PATH"'

echo "  ✓ Installing Docker dependencies..."
apt-get update
apt-get install -y docker.io docker-compose || true

echo "  ✓ Building Docker containers..."
docker-compose up --build -d

echo "  ✓ Checking container status..."
docker-compose ps

echo "  ✓ Configuring firewall..."
ufw allow 80/tcp || true
ufw allow 5000/tcp || true

echo ""
echo "=========================================="
echo "  ✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "  VeriCheck is now live at:"
echo "    http://$VPS_IP"
echo ""
echo "  Backend API:"
echo "    http://$VPS_IP:5000/api"
echo ""
ENDSSH

echo ""
echo "=========================================="
echo "  ✅ Deployment Successful!"
echo "=========================================="
echo ""
echo "  Access your application at:"
echo "    http://$VPS_IP"
echo ""
echo "  To check logs:"
echo "    ssh $VPS_USER@$VPS_IP 'docker-compose -f $VPS_PATH/docker-compose.yml logs -f'"
echo ""
echo "  To stop:"
echo "    ssh $VPS_USER@$VPS_IP 'cd $VPS_PATH && docker-compose down'"
echo ""