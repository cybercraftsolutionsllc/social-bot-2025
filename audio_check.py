import sounddevice as sd, numpy as np, time, sys
print("=== Output devices ===")
devices = sd.query_devices()
outs = [i for i,d in enumerate(devices) if d['max_output_channels']>0]
for i in outs:
    d = devices[i]
    print(f"[{i}] {d['name']}  ({d.get('hostapi','?')})")
print("\nPlaying a 440Hz tone for ~0.6s on each output device...\n")
sr = 48000
t = np.linspace(0,0.6,int(sr*0.6),False)
tone = (0.3*np.sin(2*np.pi*440*t)).astype(np.float32)
heard = []
for i in outs:
    try:
        sd.default.device = (None, i)
        sd.play(tone, sr, blocking=True)
        print(f"Played on device [{i}] -> Did you hear it?")
        heard.append(i)
        time.sleep(0.2)
    except Exception as e:
        print(f"Device [{i}] failed: {e}")
print("\nIf you heard one, pick that device index and set AUDIO_OUT_ID to it.")
if heard:
    print("Candidates that played without error:", heard)
