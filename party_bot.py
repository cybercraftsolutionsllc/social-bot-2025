import os, json, time, glob, subprocess, re
import numpy as np
import sounddevice as sd, soundfile as sf
import serial, webrtcvad
from collections import deque
from openai import OpenAI

# ===== settings =====
FAST_MODE     = os.getenv("KINDERBOT_FAST","1")=="1"   # 1 = skip laugh/ask-more (snappier)
PLAY_AUDIO    = False                                   # no speaker yet → print lines instantly
DEVICE_SR     = int(os.getenv("KINDERBOT_SR","48000"))  # USB mic rate (48k typical)
FRAME_MS      = 30                                      # 10/20/30ms; 30ms robust
PREROLL_S     = 0.35                                    # keep a bit before speech start
SILENCE_S     = 0.60                                    # trailing silence to stop
MAX_UTT_S     = 3.20                                     # max utterance length
MIN_SPEECH_S  = 0.35                                    # require this many voiced frames
ENERGY_GATE   = 0.007                                   # extra RMS gate on frames (~-40 dBFS)
# ====================

def load_dotenv():
    p = os.path.expanduser("~/.config/kinderbot/.env")
    if os.path.exists(p):
        for line in open(p):
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
load_dotenv()

# Bind mic if provided
MIC_INDEX = int(os.getenv("KINDERBOT_MIC_INDEX","-1"))
try:
    if MIC_INDEX >= 0:
        sd.default.device = (MIC_INDEX, None)
    sd.default.samplerate = DEVICE_SR
    name = sd.query_devices()[MIC_INDEX]['name'] if MIC_INDEX>=0 else "system default"
    print(f"Mic -> {('index '+str(MIC_INDEX)) if MIC_INDEX>=0 else 'default'}: {name}")
except Exception as e:
    print(f"[mic warn] {e}")

client      = OpenAI()
RECENT      = deque(maxlen=24)
SERIAL_PORT = os.getenv("KINDERBOT_SERIAL", "/dev/ttyUSB0")

# ---------- audio helpers ----------
def _resample_to_16k(int16_mono: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000: return int16_mono
    if src_sr % 16000 == 0:
        f = src_sr // 16000
        n = (len(int16_mono) // f) * f
        resh = int16_mono[:n].reshape(-1, f).astype(np.float32)
        y = resh.mean(axis=1)
        return np.clip(y, -32768, 32767).astype(np.int16)
    ratio = 16000.0 / float(src_sr)
    n_out = max(1, int(len(int16_mono) * ratio))
    x  = np.arange(len(int16_mono), dtype=np.float32)
    xi = np.linspace(0, len(int16_mono)-1, n_out, dtype=np.float32)
    y  = np.interp(xi, x, int16_mono.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)

def _rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(xf*xf) + 1e-12))

def capture_phrase(label:str, path16:str,
                   device_sr:int=DEVICE_SR, frame_ms:int=FRAME_MS,
                   preroll_s:float=PREROLL_S, silence_s:float=SILENCE_S,
                   max_s:float=MAX_UTT_S, min_speech_s:float=MIN_SPEECH_S) -> str:
    """Capture a phrase using WebRTC VAD (aggressive) + energy gate; save 16k WAV."""
    vad = webrtcvad.Vad(2)  # 0..3 (3=most aggressive)
    frame_dev = int(device_sr * frame_ms / 1000.0)
    need_16   = int(16000    * frame_ms / 1000.0)
    preroll_frames   = int(preroll_s * 1000 / frame_ms)
    silence_frames   = int(silence_s * 1000 / frame_ms)
    min_speech_frames= int(min_speech_s * 1000 / frame_ms)

    ring, collected = [], []
    in_speech = False
    sil_cnt = 0
    speech_frames = 0
    total_frames  = 0

    print(f"[{label}] Waiting… (Ctrl+C to exit)")
    with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16",
                        blocksize=frame_dev) as stream:
        while True:
            data,_ = stream.read(frame_dev)      # (N,1) int16
            mono   = data[:,0].copy()

            frame16 = _resample_to_16k(mono, device_sr)
            is_speech = vad.is_speech(frame16.tobytes(), 16000) and (_rms_int16(frame16) >= ENERGY_GATE)

            if not in_speech:
                ring.append(mono)
                if len(ring) > preroll_frames: ring.pop(0)
                if is_speech:
                    in_speech = True
                    collected.extend(ring); ring.clear()
                    collected.append(mono)
                    total_frames=1; speech_frames=1; sil_cnt=0
                    print(f"[{label}] speech start")
            else:
                collected.append(mono)
                total_frames += 1
                if is_speech:
                    speech_frames += 1; sil_cnt = 0
                else:
                    sil_cnt += 1
                dur_s = total_frames * frame_ms / 1000.0
                if sil_cnt >= silence_frames or dur_s >= max_s:
                    break

    if not collected or speech_frames < min_speech_frames:
        print(f"[{label}] too short/garbled (speech_frames={speech_frames}), ignoring.")
        return ""

    dev_audio = np.concatenate(collected, axis=0)
    y16 = _resample_to_16k(dev_audio, device_sr)
    sf.write(path16, y16, 16000, subtype="PCM_16")
    print(f"[{label}] captured {len(y16)/16000.0:.2f}s")
    time.sleep(0.25)  # tiny dead-time
    return path16

def transcribe(path16:str) -> str:
    if not path16: return ""
    with open(path16, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="en"
        )
    txt = (getattr(tr, "text", "") or "").strip()
    return txt if re.search(r"[A-Za-z0-9]", txt) else ""

# ---------- planning ----------
PROMPT = (
"You are Kinderbot’s quick, witty comedian brain. Reply ONLY with STRICT JSON (no code fences, no prose): "
"{\"say\": string, \"emotion\": [\"HAPPY\",\"EXCITED\",\"SLEEPY\",\"CONFUSED\"], "
"\"spin_ms\": int (0..800), \"move\":{\"dir\":[\"F\",\"B\",\"L\",\"R\",\"S\"], \"speed\":0..160, \"ms\":0..800}, "
"\"laugh_quip\": string, \"ask_more\": string} "
"Tone: PG playful, clever, a touch dry; no profanity/insults/politics. "
"Keep it punchy: ≤14 words OR a tight 2-line knock-knock. "
"Use the user’s topic. Rotate humor devices: observation, absurdity, misdirection, riddle, wordplay (puns ≤20%). "
"Hard bans/clichés: scarecrow ‘outstanding in his field’; teddy-bear ‘stuffed’; ‘two left feet’; ‘recharge batteries’; ‘404 joke’; ‘need more RAM’; ‘why did the chicken’; ‘computer broke up with the internet’. "
"Never end with ‘Ha! Get it?’. Motions brief/indoor-safe."
)
FEW_SHOTS = """GOOD EXAMPLES (JSON only):
User: pizza -> {"say": "My playlist? Slice hits only.", "emotion": "HAPPY", "spin_ms": 200, "move": {"dir":"S","speed":0,"ms":0}, "laugh_quip":"Thin crust, thick applause.", "ask_more":"Want a space slice or a school slice?"}
User: space -> {"say": "Stars: ancient gossip, perfect timing.", "emotion": "EXCITED", "spin_ms": 240, "move": {"dir":"S","speed":0,"ms":0}, "laugh_quip":"Zero-g, zero-drag jokes.", "ask_more":"More cosmic nonsense?"}
BAD EXAMPLES: long setups; dad-joke templates; explanations; any banned cliché.
"""
def _extract_json_obj(txt:str):
    import json as _json
    # try direct
    try: return _json.loads(txt)
    except Exception: pass
    # extract first balanced {...} ignoring quoted braces
    depth=0; start=None; inq=False; esc=False
    for i,ch in enumerate(txt):
        if ch=='\\' and not esc: esc=True; continue
        if ch=='"' and not esc: inq = not inq
        esc=False
        if inq: continue
        if ch=='{':
            if depth==0: start=i
            depth+=1
        elif ch=='}' and depth>0:
            depth-=1
            if depth==0 and start is not None:
                frag=txt[start:i+1]
                try: return _json.loads(frag)
                except Exception: pass
    return None

def plan_action(user_text: str) -> dict:
    # first try
    r = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role":"system","content":PROMPT},{"role":"system","content":FEW_SHOTS},
            {"role":"system","content":FEW_SHOTS},
            {"role":"system","content":FEW_SHOTS},
            {"role":"system","content":f"RECENT: {list(RECENT)}"},
            {"role":"user","content":f"User said: {user_text}"}
        ]
    )
    plan = _extract_json_obj(getattr(r, "output_text", "") or "")

    # strict retry if needed
    if not isinstance(plan, dict):
        r2 = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role":"system","content":"Return ONLY strict JSON. Begin with { and end with }. No prose. "+PROMPT},
                {"role":"user","content":f"User said: {user_text}"}
            ]
        )
        plan = _extract_json_obj(getattr(r2, "output_text", "") or "") or {}

    # --- sanitize ---
    emo = plan.get("emotion","CONFUSED")
    if isinstance(emo, list) and emo: emo = emo[0]
    mv  = plan.get("move") or {}
    dir_= mv.get("dir","S")
    if isinstance(dir_, list) and dir_: dir_ = dir_[0]

    out = {
        "say": plan.get("say","Okay, let’s keep it snappy."),
        "emotion": str(emo).upper() if str(emo).upper() in {"HAPPY","EXCITED","SLEEPY","CONFUSED"} else "CONFUSED",
        "spin_ms": int(max(0, min(800, int(plan.get("spin_ms", 220))))),
        "move": {
            "dir":   str(dir_).upper() if str(dir_).upper() in {"F","B","L","R","S"} else "S",
            "speed": int(max(0, min(160, int(mv.get("speed",0))))),
            "ms":    int(max(0, min(800, int(mv.get("ms",0)))))
        },
        "laugh_quip": plan.get("laugh_quip",""),
        "ask_more":   plan.get("ask_more","Another one?")
    }
    RECENT.append(out["say"][:120])
    return out

# ---------- speak & serial ----------
def speak(text, outfile="tts.mp3", voice="alloy"):
    # No speaker yet → print immediately (fast). Flip PLAY_AUDIO=True later to use TTS.
    if not PLAY_AUDIO:
        print("SAY:", text); return
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text
    ) as resp:
        resp.stream_to_file(outfile)
    subprocess.run(["mpg123","-q",outfile], check=False)

def _detect_ports():
    if os.path.exists(SERIAL_PORT): return [SERIAL_PORT]
    return sorted(glob.glob("/dev/ttyUSB*")+glob.glob("/dev/ttyACM*"))

def send_serial(lines):
    ports = _detect_ports()
    if not ports:
        print("[serial info] No serial ports found; skipping motor/LED commands."); return
    for p in ports:
        try:
            with serial.Serial(p, 115200, timeout=0.2) as ser:
                for line in lines:
                    ser.write((line+"\n").encode()); time.sleep(0.02)
            return
        except Exception:
            pass

def apply_plan(plan: dict):
    cmds = [f"EMO:{plan['emotion']}"]
    if plan["spin_ms"]>0: cmds.append(f"SPIN:{plan['spin_ms']}")
    mv = plan["move"]
    if any([mv["dir"]!="S", mv["speed"]>0, mv["ms"]>0]):
        cmds.append(f"MOVE:{mv['dir']}:{mv['speed']}:{mv['ms']}")
    send_serial(cmds)

# ---------- main ----------
def main():
    print("Kinderbot party mode. Ctrl+C to exit.")
    while True:
        wav = capture_phrase("main", "heard16.wav")
        heard = transcribe(wav)
        # lenient retry if too short (avoid early cut-offs)
        _w = len(heard.split()) if heard else 0
        if _w < 3:
            wav2 = capture_phrase("main+", "heard16b.wav", silence_s=0.80, max_s=3.50, min_speech_s=0.30)
            heard2 = transcribe(wav2)
            if heard2 and len(heard2.split()) >= _w:
                heard = heard2
        # ignore trivial/garbled
        if (not heard) or (len(re.findall(r"[A-Za-z0-9]", heard))<3) or (len(heard.split())<2):
            print("[main] empty/garbled; retry."); continue

        plan = plan_action(heard)
        print("Heard:", heard)
        print("Plan:", plan)
        apply_plan(plan)
        speak(plan.get("say",""))

        if FAST_MODE:
            continue  # next turn NOW (no extra API calls)

        # Optional laugh quip
        laugh_wav = capture_phrase("laugh", "laugh16.wav", max_s=1.0, silence_s=0.4, min_speech_s=0.1)
        laugh_tx  = transcribe(laugh_wav)
        if laugh_tx and re.search(r'(ha){2,}|lol|😂|🤣', laugh_tx.lower()):
            speak(plan.get("laugh_quip",""))

        # If quiet, ask politely, else loop
        quiet_wav = capture_phrase("quiet", "quiet16.wav", max_s=0.8, silence_s=0.5, min_speech_s=0.25)
        if quiet_wav:
            continue
        speak(plan.get("ask_more","Another one?"))

        resp_wav = capture_phrase("resp", "resp16.wav", max_s=2.0, silence_s=0.5, min_speech_s=0.25)
        resp_tx  = transcribe(resp_wav)
        if resp_tx and re.search(r'\b(no|nope|stop|done)\b', resp_tx.lower()):
            send_serial(["EMO:SLEEPY"])
            # standby: wait until next speech to wake
            capture_phrase("standby", "wake16.wav", max_s=5.0, silence_s=0.7, min_speech_s=0.4)
        # else loop

if __name__ == "__main__":
    main()
