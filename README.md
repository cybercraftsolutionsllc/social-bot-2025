# Social Bot 2025 (Kinderbot party brain)

Pi-first controller that does:
- Speech-to-Text → joke/logic → Text-to-Speech
- Sends `EMO:` and `SPIN:` commands over serial to an ESP32 (TB6612 + WS2812B).
- Runs as a user systemd service and loads secrets from `~/.config/kinderbot/.env`.

## Quick start (Pi)
```bash
git clone https://github.com/YOUR-USER/social-bot-2025.git
cd social-bot-2025
bash scripts/install.sh
# Then follow the prompt to enter OPENAI_API_KEY once.
```

After install:
```bash
systemctl --user status kinderbot.service --no-pager
journalctl --user -u kinderbot -f
```

## .env
Create `~/.config/kinderbot/.env`:
```
OPENAI_API_KEY=sk-REPLACE-ME
KINDERBOT_SERIAL=/dev/ttyUSB0
```

## ESP32 firmware
See `esp32/firmware.ino` for a minimal sketch that parses serial lines `EMO:<STATE>` and `SPIN:<ms>`,
drives the TB6612, and shows colors on WS2812 LEDs.

## Notes
- Keep Pi on its USB power bank. Ensure all grounds are common between Pi (via USB ground), ESP32, TB6612, MP1584 buck, amp, and LEDs.
- If your serial port is different, update `KINDERBOT_SERIAL` in `.env`.
- Audio path: Pi 3.5mm → PAM8302 → LS1 speaker. `aplay` is used to play WAV output.
