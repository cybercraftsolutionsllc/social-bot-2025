import os, sys, time
import sounddevice as sd

def _set_output_defaults():
    dev = os.getenv("AUDIO_DEVICE")
    card = os.getenv("AUDIO_CARD")
    out_id = os.getenv("AUDIO_OUT_ID")
    ch = int(os.getenv("AUDIO_CH") or 2)
    sr = int(os.getenv("AUDIO_SR") or 48000)
    sd.default.dtype = "int16"
    sd.default.channels = ch
    sd.default.samplerate = sr
    if dev:
        sd.default.device = dev
    elif card:
        sd.default.device = (None, int(card))
    elif out_id:
        sd.default.device = (None, int(out_id))

def _try_open(rate:int, mic_index:int|None):
    try:
        # 20ms framesize at this rate
        frame = int(rate*0.02)
        with sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                            blocksize=frame, device=mic_index) as s:
            s.read(frame)
        return True
    except Exception:
        return False

def select_mic_and_rate():
    # Respect KINDERBOT_MIC_INDEX if set; else default (None)
    mic_index = int(os.getenv("KINDERBOT_MIC_INDEX","-1"))
    mic = mic_index if mic_index >= 0 else None
    # Try requested SR first (if provided), then safe fallbacks
    want = []
    if os.getenv("KINDERBOT_SR"):
        try: want.append(int(os.getenv("KINDERBOT_SR")))
        except: pass
    for r in (16000, 44100, 48000):
        if r not in want: want.append(r)
    for r in want:
        if _try_open(r, mic):
            os.environ["KINDERBOT_SR"] = str(r)
            try:
                name = sd.query_devices()[mic]['name'] if isinstance(mic,int) else "system default"
            except Exception:
                name = "system default"
            print(f"Mic bound -> index={mic if isinstance(mic,int) else 'default'} ({name}), SR={r}")
            return r, mic
    # If we got here, no mic opens—leave SR at 16000 to keep VAD sane
    os.environ["KINDERBOT_SR"] = "16000"
    sys.stderr.write("[mic warn] No input device opened; using SR=16000 and waiting.\n")
    return 16000, mic

# ---- apply on import ----
_set_output_defaults()
select_mic_and_rate()
