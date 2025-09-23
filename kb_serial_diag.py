import sys, time, glob, serial
ports = [p.strip() for p in (sys.argv[1:] or (glob.glob("/dev/ttyUSB*")+glob.glob("/dev/ttyACM*")))]
if not ports:
    print("[diag] no /dev/ttyUSB* or /dev/ttyACM* ports found")
    sys.exit(1)
cmd_sets = [
    ["PING", "EMO:PLAY", "LED OFF", "SPIN:180", "SPIN_CCW:180"],
    ["EMO PLAY", "LED OFF", "SPIN 180", "SPIN_CCW 180"],
    ["EMO:EXCITED", "SPIN:-180", "SPIN_CCW:-180", "LED OFF"],  # alt sign format
]
for p in ports:
    print(f"[diag] trying {p} ...")
    try:
        with serial.Serial(p, 115200, timeout=0.3) as ser:
            for i, cmds in enumerate(cmd_sets, 1):
                for c in cmds:
                    ser.write((c+"\n").encode()); ser.flush(); time.sleep(0.08)
                time.sleep(0.25)
            print(f"[diag] SUCCESS wrote to {p}")
    except Exception as e:
        print(f"[diag] FAIL {p}: {e}")
print("[diag] done")
