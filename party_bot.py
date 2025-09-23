import os, re, time, glob, subprocess, traceback, warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---- env ----
from dotenv import load_dotenv
DOTENV = Path.home() / ".config" / "kinderbot" / ".env"
load_dotenv(dotenv_path=DOTENV, override=True)

raw_key = os.getenv("OPENAI_API_KEY", "").strip()
if not raw_key or not raw_key.startswith(("sk-","sk-proj-")):
    raise RuntimeError(f"OPENAI_API_KEY missing/malformed (loaded from {DOTENV})")

# ---- libs ----
import numpy as np
import sounddevice as sd, soundfile as sf, webrtcvad, serial
from openai import OpenAI
from kinderbot_behaviors import MotionPlanner, no_speech_fallback, SpeechGate, should_interrupt, send_motion

# ---- settings ----
FAST_MODE     = True
PLAY_AUDIO    = True
VOICE_MODEL   = os.getenv("KINDERBOT_TTS","openai")     # "openai" or "espeak"
OPENAI_VOICE  = os.getenv("KINDERBOT_VOICE","alloy")
MODEL_PLAN    = os.getenv("KINDERBOT_LLM","gpt-5-nano")

SERIAL_PORT   = os.getenv("KINDERBOT_SERIAL","/dev/ttyUSB0")

ENERGY_GATE   = float(os.getenv("ENERGY_GATE","0.002"))
FRAME_MS      = 20
PREROLL_S     = 0.20
SILENCE_S     = 0.35
MAX_UTT_S     = 2.20
MIN_SPEECH_S  = 0.25

# ---- audio I/O binding (SINGLE source of truth) ----
def _choose_input_index():
    """Pick input device: env override -> USB-ish -> first input-capable."""
    want = os.getenv("KINDERBOT_MIC_INDEX","").strip()
    if want.lstrip("-").isdigit():
        idx = int(want)
        try:
            d = sd.query_devices()[idx]
            if d.get("max_input_channels",0)>0:
                return idx, d["name"]
        except Exception:
            pass
    # prefer USB-ish names
    prefer = ("usb","lavalier","microphone","mic","cmedia","audio")
    avoid  = ("hdmi","headphone","bcm2835","vc4")
    cand=[]
    for i,d in enumerate(sd.query_devices()):
        if d.get("max_input_channels",0)>0:
            name=d.get("name","").lower()
            score = d["max_input_channels"]
            if any(k in name for k in prefer): score += 10
            if any(k in name for k in avoid):  score -= 10
            cand.append((score,i,d["name"]))
    cand.sort(reverse=True)
    if cand:
        return cand[0][1], cand[0][2]
    return None, "system default"

def _first_openable_rate(idx):
    for r in (int(os.getenv("KINDERBOT_SR","0") or 0), 16000, 24000, 32000, 44100, 48000):
        if not r: continue
        try:
            frame=int(r*0.02)
            with sd.InputStream(samplerate=r, channels=1, dtype="int16",
                                blocksize=frame, device=idx) as s:
                s.read(frame)
            return r
        except Exception:
            continue
    return 16000

def bind_audio_once():
    # output (optional; leave alone unless env asks)
    sd.default.dtype = "int16"
    if os.getenv("AUDIO_DEVICE"):
        sd.default.device = os.getenv("AUDIO_DEVICE")
    elif os.getenv("AUDIO_CARD"):
        sd.default.device = (None, int(os.getenv("AUDIO_CARD")))
    elif os.getenv("AUDIO_OUT_ID"):
        sd.default.device = (None, int(os.getenv("AUDIO_OUT_ID")))

    # input: pick, test, and bind ONLY the input side
    in_idx, in_name = _choose_input_index()
    if in_idx is None:
        print("Mic -> NONE found; will use system default")
        sr = int(os.getenv("KINDERBOT_SR","16000"))
        sd.default.samplerate = sr
        return None, sr
    # keep existing output side if tuple
    out_side = None
    try:
        cur = sd.default.device
        if isinstance(cur,(list,tuple)) and len(cur)==2:
            out_side = cur[1]
    except Exception: pass

    sr = _first_openable_rate(in_idx)
    sd.default.device = (in_idx, out_side)     # bind INPUT ONLY
    sd.default.samplerate = sr
    os.environ["KINDERBOT_MIC_INDEX"]=str(in_idx)
    os.environ["KINDERBOT_SR"]=str(sr)
    print(f"Mic -> bound index={in_idx} name={in_name} SR={sr}")
    return in_idx, sr

MIC_INDEX, DEVICE_SR = bind_audio_once()

# ---- helpers ----
def _to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000: return x
    ratio = 16000.0 / float(src_sr)
    n_out = max(1, int(len(x) * ratio))
    xi = np.linspace(0, len(x)-1, n_out, dtype=np.float32)
    y  = np.interp(xi, np.arange(len(x), dtype=np.float32), x.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)

def _rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)/32768.0
    return float(np.sqrt(np.mean(xf*xf) + 1e-12))

def capture_phrase(label:str, path16:str,
                   device_sr:int=DEVICE_SR, frame_ms:int=FRAME_MS,
                   preroll_s:float=PREROLL_S, silence_s:float=SILENCE_S,
                   max_s:float=MAX_UTT_S, min_speech_s:float=MIN_SPEECH_S,
                   max_wait_s:float=6.0) -> str:
    """Capture one phrase from the *explicit* input index; save 16k WAV."""
    idx = MIC_INDEX if (isinstance(MIC_INDEX,int) and MIC_INDEX>=0) else None
    vad = webrtcvad.Vad(3)
    fdev = int(device_sr * frame_ms / 1000.0)
    pre_f   = int(preroll_s * 1000 / frame_ms)
    sil_f   = int(silence_s * 1000 / frame_ms)
    min_f   = int(min_speech_s * 1000 / frame_ms)

    ring, collected = [], []
    in_speech=False; sil_cnt=0; speech_f=0; total_f=0; start=time.time()

    dev_name = sd.query_devices()[idx]['name'] if isinstance(idx,int) else "system default"
    print(f"[{label}] Waiting… (Ctrl+C to exit)  (input={idx}:{dev_name}, SR={device_sr})")

    with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16",
                        blocksize=fdev, device=idx) as stream:
        while True:
            if not in_speech and (time.time()-start) > max_wait_s:
                print(f"[{label}] no speech timeout"); return ""
            data,_ = stream.read(fdev)
            mono = data[:,0]
            f16  = _to_16k(mono, device_sr)
            talking = (_rms_int16(f16) >= ENERGY_GATE) and vad.is_speech(f16.tobytes(), 16000)

            if not in_speech:
                ring.append(mono)
                if len(ring) > pre_f: ring.pop(0)
                if talking:
                    in_speech=True
                    collected.extend(ring); ring.clear()
                    collected.append(mono)
                    total_f=1; speech_f=1; sil_cnt=0
                    print(f"[{label}] speech start")
            else:
                collected.append(mono); total_f += 1
                if talking: speech_f += 1; sil_cnt = 0
                else: sil_cnt += 1
                dur_s = total_f * frame_ms / 1000.0
                if sil_cnt >= sil_f or dur_s >= max_s:
                    break

    if not collected or speech_f < min_f:
        print(f"[{label}] too short/garbled (speech_frames={speech_f}), ignoring.")
        return ""

    y16 = _to_16k(np.concatenate(collected,axis=0), device_sr)
    sf.write(path16, y16, 16000, subtype="PCM_16")
    print(f"[{label}] captured {len(y16)/16000.0:.2f}s")
    time.sleep(0.05)
    return path16

# ---- OpenAI ----
client = OpenAI(api_key=raw_key)
RECENT = deque(maxlen=24)

PROMPT = ("You are Laughbot. You interact with the user based on their input by returning funny one-liners, dad jokes, and anythign else funny. "
          "No pretext, no follow up. No need for fancy build up, just reply with some funny, entertaining jokes or zings based on input, or not if the input is not coherent.")

def transcribe(path16:str) -> str:
    if not path16: return ""
    try:
        with open(path16, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=f, language="en"
            )
        txt = (getattr(tr, "text", "") or "").strip()
        return txt if re.search(r"[A-Za-z0-9]", txt) else ""
    except Exception as e:
        print("[stt err]", e); return ""

def plan_action(user_text: str) -> dict:
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

# ---- TTS ----
def say_espeak(text):
    try: subprocess.run(["espeak-ng","-s","165","-p","45",text], check=False)
    except Exception as e: print("[espeak warn]", e)

def say_openai(text, outfile="tts.mp3", voice=OPENAI_VOICE):
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice=voice, input=text
        ) as resp:
            resp.stream_to_file(outfile)
        subprocess.run(["mpg123","-q",outfile], check=False)
    except Exception as e:
        print("[tts err]", e); say_espeak(text)

def speak(text):
    if not PLAY_AUDIO:
        print("SAY:", text); return
    (say_openai if VOICE_MODEL.lower().startswith("openai") else say_espeak)(text)

# ---- serial/motion ----
def _detect_ports():
    if os.path.exists(SERIAL_PORT): return [SERIAL_PORT]
    return sorted(glob.glob("/dev/ttyUSB*")+glob.glob("/dev/ttyACM*"))

def send_serial(lines):
    ports = _detect_ports()
    if not ports:
        print("[serial] no ports; skipping:", lines); return
    for p in ports:
        try:
            with serial.Serial(p, 115200, timeout=0.2) as ser:
                for line in lines:
                    ser.write((line+"\n").encode()); time.sleep(0.02)
            return
        except Exception as e:
            print(f"[serial warn] {p}: {e}")

planner = MotionPlanner(home_every=(5,10))
def apply_plan(plan: dict):
    send_motion(lambda s: send_serial([s.strip()]), planner.next_move())

# ---- main ----
def main():
    print("Kinderbot party mode. Ctrl+C to exit.")
    while True:
        try:
            wav = capture_phrase("main", "heard16.wav", device_sr=int(os.getenv("KINDERBOT_SR","16000")))
            heard = transcribe(wav)
            if not heard or len(re.findall(r"[A-Za-z0-9]", heard))<3 or len(heard.split())<2:
                print("[main] empty/garbled; fallback")
                speak(no_speech_fallback())
                continue

            plan = plan_action(heard)
            print("Heard:", heard)
            print("Line:", plan["say"])

            apply_plan(plan)
            speak(plan.get("say",""))

            if FAST_MODE: continue

        except KeyboardInterrupt:
            print("\n[exit]"); break
        except Exception as e:
            print("[loop err]", e); traceback.print_exc(); time.sleep(0.3)

if __name__ == "__main__":
    main()
