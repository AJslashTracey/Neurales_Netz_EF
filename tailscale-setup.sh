#!/usr/bin/env bash
# Tailscale setup for Fedora - make this machine server-ready
# Run with: sudo ./tailscale-setup.sh
#
# For headless/server auth (no browser):
#   sudo AUTHKEY=tskey-auth-xxx ./tailscale-setup.sh

set -e

echo "=== Tailscale Server Setup for Fedora ==="

# 1. Install Tailscale (official method)
if ! command -v tailscale &>/dev/null; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
else
    echo "Tailscale already installed."
fi

# 2. Enable and start Tailscale service
echo "Enabling Tailscale service..."
systemctl enable tailscaled
systemctl start tailscaled

# 3. Connect to Tailscale
echo ""
if [[ -n "${AUTHKEY:-}" ]]; then
    echo "Connecting with auth key (headless mode)..."
    sudo tailscale up --authkey="$AUTHKEY" --accept-routes
else
    echo "Connecting to Tailscale network..."
    echo "A browser window will open for authentication."
    echo ""
    echo "For headless/server setup: AUTHKEY=tskey-auth-xxx sudo ./tailscale-setup.sh"
    echo "To disable key expiry (recommended for servers):"
    echo "  Admin console -> Machines -> [this machine] -> Disable key expiry"
    echo ""
    sudo tailscale up
fi

# 4. Show Tailscale IP
echo ""
echo "=== Tailscale IP (use this to reach this machine) ==="
tailscale ip -4 2>/dev/null || tailscale ip 2>/dev/null || echo "Run 'tailscale ip -4' after auth completes."

# 5. Optional: ensure SSH is enabled for remote access
if systemctl is-enabled sshd &>/dev/null; then
    echo ""
    echo "SSH (sshd) is enabled. Connect via: ssh user@$(tailscale ip -4 2>/dev/null || echo 'TAILSCALE_IP')"
elif systemctl is-enabled ssh &>/dev/null; then
    echo ""
    echo "SSH (ssh) is enabled."
else
    echo ""
    echo "Tip: Enable SSH for remote access: sudo systemctl enable --now sshd"
fi

echo ""
echo "=== Setup complete. This machine is now on your Tailscale network. ==="
