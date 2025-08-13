import os, time, sys, serial
port = os.getenv("KINDERBOT_SERIAL", "/dev/ttyUSB0")
ser = serial.Serial(port, 115200, timeout=0.5)
for line in ["EMO:EXCITED", "SPIN:700", "MOVE:F:180:500"]:
    ser.write((line+"\n").encode())
    time.sleep(0.05)
print("Sent EMO/ SPIN/ MOVE to", port)
