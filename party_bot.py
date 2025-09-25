import os, re, time, glob, json, subprocess, warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ── env / keys ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
DOTENV = Path.home() / ".config" / "kinderbot" / ".env"
load_dotenv(dotenv_path=DOTENV, override=True)

RAW_KEY = (os.getenv("OPENAI_API_KEY","") or "").strip()
if not RAW_KEY or not RAW_KEY.startswith(("sk-","sk-proj-")):
    raise RuntimeError(f"OPENAI_API_KEY missing/malformed (loaded from {DOTENV})")

# ── audio libs / io ────────────────────────────────────────────────────────────
import numpy as np
import sounddevice as sd, soundfile as sf, webrtcvad
# ── serial ─────────────────────────────────────────────────────────────────────
import serial
# ── openai ─────────────────────────────────────────────────────────────────────
from openai import OpenAI

# ── behaviors (keep your existing helpers) ─────────────────────────────────────
from kinderbot_behaviors import no_speech_fallback

# ── settings (kept yours; prompt unchanged) ───────────────────────────────────
FAST_MODE     = True
PLAY_AUDIO    = True
VOICE_MODEL   = os.getenv("KINDERBOT_TTS","openai")
OPENAI_VOICE  = os.getenv("KINDERBOT_VOICE","alloy")
MODEL_PLAN    = os.getenv("KINDERBOT_LLM","gpt-5-nano")
SERIAL_PORT   = os.getenv("KINDERBOT_SERIAL","/dev/ttyUSB0")

ENERGY_GATE   = 0.0015
FRAME_MS      = 20
PREROLL_S     = 0.20
SILENCE_S     = 0.60
MAX_UTT_S     = 8.00
MIN_SPEECH_S  = 0.18

PROMPT = ("You are Laughbot, you are a bot replacement for Rodney Dangerfield. Reply with ONE short, clean, clever line.No lists, no bullets, no emojis, no quotes, no meta commentary "
          "You interact with the user based on their input by returning funny one-liners, dad jokes, and anything else funny. "
          "No pretext, no follow up. No need for fancy build up, just reply with some funny, entertaining jokes or zings based on input, or not if the input is not coherent.")

# ── mic binding (single source of truth) ───────────────────────────────────────
def _choose_input_index():
    want = (os.getenv("KINDERBOT_MIC_INDEX","") or "").strip()
    if want.lstrip("-").isdigit():
        idx = int(want)
        try:
            d = sd.query_devices()[idx]
            if d.get("max_input_channels",0)>0: return idx, d["name"]
        except Exception: pass
    prefer, avoid = ("usb","lavalier","microphone","mic","cmedia","audio"), ("hdmi","headphone","bcm2835","vc4")
    cand=[]
    for i,d in enumerate(sd.query_devices()):
        if d.get("max_input_channels",0)>0:
            n=d.get("name","").lower(); score=d["max_input_channels"] + (10 if any(k in n for k in prefer) else 0) - (10 if any(k in n for k in avoid) else 0)
            cand.append((score,i,d.get("name","")))
    cand.sort(reverse=True)
    return (cand[0][1], cand[0][2]) if cand else (None,"system default")

def _first_openable_rate(idx):
    for r in (int(os.getenv("KINDERBOT_SR","0") or 0), 44100, 48000, 32000, 16000):
        if not r: continue
        try:
            frame=int(r*0.02)
            with sd.InputStream(samplerate=r, channels=1, dtype="int16", blocksize=frame, device=idx) as s:
                s.read(frame)
            return r
        except Exception: continue
    return 16000

def bind_audio_once():
    sd.default.dtype = "int16"
    in_idx, in_name = _choose_input_index()
    if in_idx is None:
        sr = int(os.getenv("KINDERBOT_SR","16000")); sd.default.samplerate = sr
        print(f"Mic -> bound index=default (system default), SR={sr}")
        return None, sr
    # keep output side if previously set
    out_side = None
    try:
        cur = sd.default.device
        if isinstance(cur,(list,tuple)) and len(cur)==2: out_side = cur[1]
    except Exception: pass
    sr = _first_openable_rate(in_idx)
    sd.default.device = (in_idx, out_side)
    sd.default.samplerate = sr
    os.environ["KINDERBOT_MIC_INDEX"]=str(in_idx); os.environ["KINDERBOT_SR"]=str(sr)
    print(f"Mic -> bound index={in_idx} name={in_name} SR={sr}")
    return in_idx, sr

MIC_INDEX, DEVICE_SR = bind_audio_once()

# ── capture / VAD ──────────────────────────────────────────────────────────────
def _to_16k(x: np.ndarray, src_sr:int)->np.ndarray:
    if src_sr == 16000: return x
    ratio = 16000.0/float(src_sr)
    n_out = max(1, int(len(x)*ratio))
    xi = np.linspace(0, len(x)-1, n_out, dtype=np.float32)
    y  = np.interp(xi, np.arange(len(x), dtype=np.float32), x.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)

def _rms_int16(x: np.ndarray)->float:
    xf = x.astype(np.float32)/32768.0
    return float(np.sqrt(np.mean(xf*xf) + 1e-12))

def capture_phrase(label:str, path16:str,
                   device_sr:int=DEVICE_SR, frame_ms:int=FRAME_MS,
                   preroll_s:float=PREROLL_S, silence_s:float=SILENCE_S,
                   max_s:float=MAX_UTT_S, min_speech_s:float=MIN_SPEECH_S,
                   max_wait_s:float=6.0)->str:
    idx = MIC_INDEX if (isinstance(MIC_INDEX,int) and MIC_INDEX>=0) else None
    vad = webrtcvad.Vad(3)
    fdev = int(device_sr * frame_ms / 1000.0)
    pre_f, sil_f, min_f = int(preroll_s*1000/FRAME_MS), int(silence_s*1000/FRAME_MS), int(min_speech_s*1000/FRAME_MS)
    ring, collected = [], []
    in_speech=False; sil_cnt=0; speech_f=0; total_f=0; start=time.time()
    name = sd.query_devices()[idx]['name'] if isinstance(idx,int) else "system default"
    print(f"[{label}] Waiting… (Ctrl+C to exit)  (input={idx}:{name}, SR={device_sr})")
    with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16", blocksize=fdev, device=idx) as stream:
        while True:
            if not in_speech and (time.time()-start)>max_wait_s:
                print(f"[{label}] no speech timeout"); return ""
            data,_ = stream.read(fdev)
            mono = data[:,0]; f16 = _to_16k(mono, device_sr)
            talking = (_rms_int16(f16) >= ENERGY_GATE) and vad.is_speech(f16.tobytes(), 16000)
            if not in_speech:
                ring.append(mono)
                if len(ring)>pre_f: ring.pop(0)
                if talking:
                    in_speech=True; collected.extend(ring); ring.clear()
                    collected.append(mono); total_f=1; speech_f=1; sil_cnt=0
                    print(f"[{label}] speech start")
            else:
                collected.append(mono); total_f += 1
                if talking: speech_f += 1; sil_cnt = 0
                else: sil_cnt += 1
                dur_s = total_f * frame_ms / 1000.0
                if (sil_cnt>=sil_f and speech_f>=min_f) or dur_s>=max_s:
                    break
    if not collected or speech_f<min_f:
        print(f"[{label}] too short/garbled (speech_frames={speech_f}), ignoring."); return ""
    y16 = _to_16k(np.concatenate(collected,axis=0), device_sr)
    sf.write(path16, y16, 16000, subtype="PCM_16")
    print(f"[{label}] captured {len(y16)/16000.0:.2f}s"); time.sleep(0.05)
    return path16

# ── OpenAI helpers (prompt unchanged) ──────────────────────────────────────────
client = OpenAI(api_key=RAW_KEY)
RECENT = deque(maxlen=24)

def transcribe(path16:str)->str:
    if not path16: return ""
    try:
        with open(path16, "rb") as f:
            tr = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f, language="en")
        txt = (getattr(tr,"text","") or "").strip()
        return txt if re.search(r"[A-Za-z0-9]", txt) else ""
    except Exception as e:
        print("[stt err]", e); return ""

def plan_action(user_text:str)->dict:
    try:
        r = client.responses.create(
            model=MODEL_PLAN,
            input=[{"role":"system","content":PROMPT},
                   {"role":"user","content":f"User said: {user_text}"}]
        )
        line = (getattr(r,"output_text","") or "").strip()
    except Exception as e:
        print("[llm err]", e); line = "Okay, let’s keep it snappy."
    RECENT.append(line[:120])
    return {"say": line}

# ── TTS ────────────────────────────────────────────────────────────────────────
def say_espeak(text):
    try:
        import os, shlex, subprocess
        dev = os.getenv("KINDERBOT_SPKR","")
        if dev:
            # render to stdout and send to ALSA device explicitly
            cmd = f'espeak-ng -s 165 -p 45 --stdout {shlex.quote(text)} | aplay -q -D {shlex.quote(dev)}'
            subprocess.run(cmd, shell=True, check=False)
        else:
            subprocess.run(["espeak-ng","-s","165","-p","45",text], check=False)
    except Exception as e:
        print("[espeak warn]", e)

def say_openai(text, outfile="tts.mp3", voice=OPENAI_VOICE):
    try:
        import os
        dev = os.getenv("KINDERBOT_SPKR","")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice=voice, input=text
        ) as resp:
            resp.stream_to_file(outfile)
        cmd = ["mpg123","-q"] + (["-a", dev] if dev else []) + [outfile]
        subprocess.run(cmd, check=False)
    except Exception as e:
        print("[tts err]", e); say_espeak(text)

def speak(text):
    if not PLAY_AUDIO: print("SAY:", text); return
    (say_openai if VOICE_MODEL.lower().startswith("openai") else say_espeak)(text)

# ── serial ─────────────────────────────────────────────────────────────────────
def _detect_ports():
    return [SERIAL_PORT] if os.path.exists(SERIAL_PORT) else sorted(glob.glob("/dev/ttyUSB*")+glob.glob("/dev/ttyACM*"))

def send_serial(lines):
    ports = _detect_ports()
    if not ports:
        print("[serial] no ports; skipping:", lines); return
    tried=set(); ok=False
    for pth in ports:
        if pth in tried: continue
        tried.add(pth)
        try:
            with serial.Serial(pth, 115200, timeout=0.3) as ser:
                for line in lines:
                    ser.write((line+"\r\n").encode()); ser.flush(); time.sleep(0.12)
            print(f"[serial] wrote {len(lines)} cmds to {pth}")
            ok=True; break
        except Exception as e:
            print(f"[serial warn] {pth}: {e}")
    if not ok:
        print("[serial] no port accepted writes; cmds=", lines, "candidates=", ports)

# ── movement / LEDs (strict alternation + batching + LLM override) ────────────
MOVE_N = 0  # in-process alternator

def _dualize(cmds):
    out=[]
    for c in cmds:
        out.append(c)
        if ":" in c: out.append(c.replace(":", " ", 1))
        elif " " in c:
            head,*rest = c.split(" ",1)
            out.append(head+":"+(rest[0] if rest else ""))
    return out

def _build_bundles(direction:str, home_wiggle:bool):
    led_on  = _dualize(["LED 255 255 255","LED ON","LED:ON","RGB 255 255 255"])
    led_off = _dualize(["LED 0 0 0","LED OFF","LED:OFF","RGB 0 0 0"])
    if home_wiggle:
        turn = _dualize(["SPIN_CCW 140","SPIN 140"])  # short recenter wiggle
    else:
        if direction == "ccw":
            turn = _dualize(["SPIN -260","SPIN:-260","LEFT 260","TURN -260","CCW 260","SPIN_CCW 260","SPIN_CCW:260"])
        else:
            turn = _dualize(["SPIN 260","SPIN:260","RIGHT 260","TURN 260","CW 260"])
    return led_on, turn, led_off

def _parse_move_led_overrides(text:str):
    """
    Parses lines like:
      [MOVE cw] [LED on]
    Returns (dir_or_None, led_on_bool_or_None)
    """
    m = re.search(r'\[MOVE\s+(cw|ccw|home)\]\s*\[LED\s+(on|off)\]', text, re.I)
    if not m: return None, None
    md, ml = m.group(1).lower(), m.group(2).lower()
    return md, (ml=="on")


MOVE_N = 0  # in-process alternation counter



def apply_plan(plan: dict):
    """
    LED blink + visible move:
      - Blink: EXCITED, pause, CONFUSED, pause, EXCITED
      - Path:  FWD(long) -> LEFT(turn) -> BACK(almost long)
      - Coda:  CONFUSED
    Uses firmware grammar only: EMO:<NAME>, MOVE:<dir>:<speed>:<ms>
    Tunable via ENV:
      KINDERBOT_SPEED (default 200)
      KINDERBOT_MS_FWD (default 700)
      KINDERBOT_MS_TURN (default 350)
      KINDERBOT_MS_BACK (default 620)
      KINDERBOT_MS_BLINK (default 180)   # pause duration for LED blinks
    """
    import os, json
    SPEED = int(os.getenv("KINDERBOT_SPEED","200"))
    MS_F  = int(os.getenv("KINDERBOT_MS_FWD","700"))
    MS_T  = int(os.getenv("KINDERBOT_MS_TURN","350"))
    MS_B  = int(os.getenv("KINDERBOT_MS_BACK","620"))
    MS_BL = int(os.getenv("KINDERBOT_MS_BLINK","180"))

    cmds = [
        # LED pre-blink (create pauses using MOVE:S:0:<ms>)
        "EMO:EXCITED",           f"MOVE:S:0:{MS_BL}",
        "EMO:CONFUSED",          f"MOVE:S:0:{MS_BL}",
        "EMO:EXCITED",

        # Visible motion (step out & return with rotation)
        f"MOVE:F:{SPEED}:{MS_F}",
        f"MOVE:L:{SPEED}:{MS_T}",
        f"MOVE:B:{SPEED}:{MS_B}",

        # LED coda
        "EMO:CONFUSED"
    ]

    print(f"[motion] step-out F:{MS_F} L:{MS_T} B:{MS_B}  speed={SPEED}  blink={MS_BL}")
    send_serial(cmds)

    try: json.dump({"n": 0}, open("/tmp/kb_motion_state.json","w"))
    except Exception: pass

def _kb_selftest():
    try:
        import os
        SWAP  = os.getenv("KINDERBOT_SWAP_DIR","0").lower() in ("1","true","yes","on")
        cw  = "L" if SWAP else "R"
        ccw = "R" if SWAP else "L"
        print("[boot] self-test: EMO + CW then CCW (swap=" + str(SWAP) + ")")
        send_serial(["EMO:EXCITED", f"MOVE:{cw}:180:350"])
        send_serial([f"MOVE:{ccw}:180:350", "EMO:CONFUSED"])
    except Exception as e:
        print("[boot warn]", e)

def main():
    print("Kinderbot party mode. Ctrl+C to exit.")
    while True:
        try:
            wav = capture_phrase("main", "heard16.wav", device_sr=int(os.getenv("KINDERBOT_SR","16000")))
            heard = transcribe(wav)

            if not heard or len(re.findall(r"[A-Za-z0-9]", heard))<3 or len(heard.split())<2:
                print("[main] empty/garbled; fallback")
                # quip + joke even on silence
                quip = "Couldn't hear you—speaking up helps."
                speak(quip)
                joke = plan_action("tell a short joke")["say"]
                print("Line:", joke); apply_plan({"say": joke}); speak(joke)
                continue

            plan = plan_action(heard)
            print("Heard:", heard); print("Line:", plan["say"])
            apply_plan(plan); speak(plan.get("say",""))

            if FAST_MODE: continue

        except KeyboardInterrupt:
            print("\n[exit]"); break
        except Exception as e:
            import traceback; print("[loop err]", e); traceback.print_exc(); time.sleep(0.3)

# ── guard (AFTER definitions) ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("[boot] entering main()")
    _kb_selftest()
    main()
