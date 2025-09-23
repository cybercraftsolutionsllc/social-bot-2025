import os, sounddevice as sd
def force_bind_input():
    v = os.getenv("KINDERBOT_MIC_INDEX","-1")
    try: idx = int(v)
    except: idx = -1
    if idx < 0:
        # fallback: first input-capable device
        for i,d in enumerate(sd.query_devices()):
            if d.get("max_input_channels",0)>0:
                idx = i; break
    # keep current output side if present
    out_dev = None
    try:
        cur = sd.default.device
        if isinstance(cur,(list,tuple)) and len(cur)==2:
            out_dev = cur[1]
    except Exception:
        pass
    sd.default.device = (idx, out_dev)
    # honor KINDERBOT_SR if set
    try:
        sr = int(os.getenv("KINDERBOT_SR","16000"))
        sd.default.samplerate = sr
    except Exception:
        pass
    try:
        name = sd.query_devices()[idx]['name']
    except Exception:
        name = "unknown"
    print(f"Mic FORCE bind -> index={idx} name={name} (out={out_dev})")
