import os, sounddevice as sd

def pick_input_index():
    prefer = ("usb", "microphone", "mic", "cmedia", "audio")
    avoid  = ("hdmi", "headphone", "bcm2835", "hdmi", "vc4")
    cand = []
    for i,d in enumerate(sd.query_devices()):
        if d.get("max_input_channels",0) > 0:
            name = d.get("name","").lower()
            score = 0
            if any(k in name for k in prefer): score += 10
            if any(k in name for k in avoid):  score -= 10
            score += d["max_input_channels"]
            cand.append((score, i, d["name"]))
    cand.sort(reverse=True)
    return cand[0][1] if cand else None, (cand[0][2] if cand else "none")

def first_openable_rate(idx):
    for r in (16000, 24000, 32000, 44100, 48000):
        try:
            frame = int(r*0.02)
            with sd.InputStream(samplerate=r, channels=1, dtype="int16",
                                blocksize=frame, device=idx) as s:
                s.read(frame)
            return r
        except Exception:
            continue
    return 16000

# ---- bind input side only ----
idx, name = pick_input_index()
out_dev = None
try:
    cur = sd.default.device
    # cur may be None, int, str, or (in_idx, out_idx)
    if isinstance(cur, (list, tuple)) and len(cur)==2:
        out_dev = cur[1]
except Exception:
    pass

if idx is not None:
    sd.default.device = (idx, out_dev)  # bind *input* only, keep output as-is
    os.environ["KINDERBOT_MIC_INDEX"] = str(idx)
    sr = first_openable_rate(idx)
    os.environ["KINDERBOT_SR"] = str(sr)
    print(f"Mic bind -> index={idx} ({name}), SR={sr}")
else:
    os.environ.setdefault("KINDERBOT_SR","16000")
    print("Mic bind -> none found; using system default, SR=", os.environ["KINDERBOT_SR"])
