#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/kinderbot"
ENV_DIR="$HOME/.config/kinderbot"
ENV_FILE="$ENV_DIR/.env"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_NAME="kinderbot.service"

echo "[1/6] Installing system packages..."
sudo apt update -y
sudo apt install -y python3-venv python3-dev portaudio19-dev libsndfile1 alsa-utils

echo "[2/6] Creating virtualenv at $VENV ..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"

echo "[3/6] Preparing env file..."
mkdir -p "$ENV_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$APP_DIR/.env.example" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "An .env was created at $ENV_FILE"
  read -p "Enter your OPENAI_API_KEY (sk-...): " KEY
  if [[ -n "${KEY:-}" ]]; then
    sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=${KEY}|" "$ENV_FILE"
  fi
  read -p "Serial device [/dev/ttyUSB0]: " PORT
  PORT="${PORT:-/dev/ttyUSB0}"
  sed -i "s|KINDERBOT_SERIAL=.*|KINDERBOT_SERIAL=${PORT}|" "$ENV_FILE"
fi

echo "[4/6] Installing systemd user service..."
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/$UNIT_NAME" <<'UNIT'
[Unit]
Description=Kinderbot party brain
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/kinderbot/.env
WorkingDirectory=%h/social-bot-2025
ExecStart=%h/kinderbot/bin/python %h/social-bot-2025/party_bot.py
Restart=on-failure
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
UNIT

echo "[5/6] Enabling linger so user services run after reboot..."
sudo loginctl enable-linger "$USER" || true

echo "[6/6] Enabling + starting service..."
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT_NAME"

echo "Done! Check status:"
echo "  systemctl --user status $UNIT_NAME --no-pager"
echo "  journalctl --user -u $UNIT_NAME -f"
