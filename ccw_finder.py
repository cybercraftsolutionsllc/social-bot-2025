import time, os, serial
PORT = os.getenv("KINDERBOT_SERIAL","/dev/ttyUSB0")
CANDIDATES = [
    "SPIN_CCW 260","SPIN_CCW:260",
    "SPIN -260","SPIN:-260",
    "TURN -260","TURN:-260",
    "LEFT 260","LEFT:260",
    "CCW 260","CCW:260",
    "TURNL 260","TURNL:260",
    "ROTATE_L 260","ROTATE_L:260",
    "SPIN 260 DIR CCW","SPIN:260 DIR:CCW",
]
ALT = []
for c in CANDIDATES:
    ALT.append(c)
    if ":" in c: ALT.append(c.replace(":", " ", 1))
    elif " " in c:
        head,*rest = c.split(" ",1); ALT.append(head+":"+(rest[0] if rest else ""))
CANDIDATES = ALT

print(f"[probe] port={PORT} CCW candidates={len(CANDIDATES)}")
with serial.Serial(PORT,115200,timeout=0.3) as ser:
    for i,c in enumerate(CANDIDATES,1):
        print(f"[{i:02d}] {c}")
        ser.write((c+"\r\n").encode()); ser.flush()
        time.sleep(0.6)
        # brief stop between tries
        ser.write(("LED 0 0 0\r\n").encode()); ser.flush()
        time.sleep(0.2)
print("[probe] done. Tell me the FIRST index that actually turned CCW.")
