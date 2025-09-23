import os, time, serial, sys
PORT = os.getenv("KINDERBOT_SERIAL","/dev/ttyUSB0")
BAUD = 115200
HELPS = ["HELP","help","?","H","CMDS","COMMANDS","LIST","INFO","VER","VERSION"]
print(f"[ident] opening {PORT} @ {BAUD}")
with serial.Serial(PORT, BAUD, timeout=0.3) as ser:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    for h in HELPS:
        print(f"[ident] >>> {h}")
        ser.write((h+"\r\n").encode()); ser.flush()
        t0=time.time()
        while time.time()-t0 < 1.5:
            try:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    print("[ident] <<<", line)
            except Exception: break
    print("[ident] done")
