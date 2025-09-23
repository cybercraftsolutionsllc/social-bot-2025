import sounddevice as sd
try:
    dev = sd.default.device
    in_idx = dev[0] if isinstance(dev,(list,tuple)) else dev
    name = sd.query_devices()[in_idx]['name'] if isinstance(in_idx,int) and in_idx is not None else "system default"
    print(f"Mic -> bound index={in_idx} name={name}")
except Exception as e:
    print(f"[mic warn] bind check: {e}")
