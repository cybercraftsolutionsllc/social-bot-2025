import os, json, time, subprocess, sys, pathlib
import sounddevice as sd, soundfile as sf
import serial
from openai import OpenAI

# Load env from ~/.config/kinderbot/.env if present
def load_dotenv():
    env_path = os.path.expanduser("~/.config/kinderbot/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#") or "=" not in line: 
                    continue
                k,v = line.split("=",1)
                os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

client = OpenAI()
SERIAL_PORT = os.getenv("KINDERBOT_SERIAL", "/dev/ttyUSB0")

def record_wav(path="in.wav", seconds=3, sr=16000):
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    sf.write(path, audio, sr, subtype="PCM_16")
    return path

def transcribe(path):
    with open(path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return (tr.text or "").strip()

def plan_joke(user_text:str):
    system = (
        "You are Kinderbot's kid-friendly comedian brain. "
        "Make a very short one-line joke or quip, then choose an LED emotion "
        "from [HAPPY, EXCITED, SLEEPY, CONFUSED], and a spin duration 0–1500 ms. "
        "Return ONLY JSON like: {\"say\": str, \"emotion\": str, \"spin_ms\": int}."
    )
    resp = client.responses.create(
        model="o4-mini",
        input=[{"role":"system","content":system},
               {"role":"user","content":f"User said: {user_text}"}]
    )
    try:
        return json.loads(resp.output_text)
    except Exception as e:
        # Fallback: say something safe
        return {"say":"I'm having a silly day!", "emotion":"HAPPY", "spin_ms":500}

def speak(text, outfile="tts.wav", voice="alloy"):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        format="wav"
    )
    with open(outfile, "wb") as f:
        f.write(speech.read())
    subprocess.run(["aplay", "-q", outfile], check=False)

def send_serial(lines):
    try:
        with serial.Serial(SERIAL_PORT, 115200, timeout=0.2) as ser:
            for line in lines:
                ser.write((line+"\n").encode("utf-8"))
                time.sleep(0.02)
    except Exception as e:
        print(f"[serial warn] {e}")

def main():
    print("Kinderbot party mode. Ctrl+C to exit.")
    while True:
        wav = record_wav(seconds=3)
        heard = transcribe(wav)
        if not heard:
            continue
        plan = plan_joke(heard)
        print("Heard:", heard)
        print("Plan:", plan)
        send_serial([f"EMO:{plan['emotion']}", f"SPIN:{int(plan['spin_ms'])}"])
        speak(plan["say"])

if __name__ == "__main__":
    main()
