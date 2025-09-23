import os, time
import numpy as np, sounddevice as sd, soundfile as sf, webrtcvad

ENERGY_GATE = float(os.getenv("ENERGY_GATE","0.002"))
FRAME_MS    = 20

def _rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(xf*xf) + 1e-12))

def _to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000: return x
    ratio = 16000.0 / float(src_sr)
    n_out = max(1, int(len(x) * ratio))
    xi = np.linspace(0, len(x)-1, n_out, dtype=np.float32)
    y  = np.interp(xi, np.arange(len(x), dtype=np.float32), x.astype(np.float32))
    return np.clip(y, -32768, 32767).astype(np.int16)

def capture_phrase(label:str, path16:str,
                   device_sr:int=int(os.getenv("KINDERBOT_SR","16000")),
                   preroll_s:float=0.20, silence_s:float=0.35,
                   max_s:float=2.20, min_speech_s:float=0.25,
                   max_wait_s:float=6.0) -> str:
    vad = webrtcvad.Vad(3)
    fdev = int(device_sr * FRAME_MS / 1000.0)
    pre_f   = int(preroll_s * 1000 / FRAME_MS)
    sil_f   = int(silence_s * 1000 / FRAME_MS)
    min_f   = int(min_speech_s * 1000 / FRAME_MS)

    ring, collected = [], []
    in_speech = False
    sil_cnt = 0
    speech_frames = 0
    total_frames  = 0
    start = time.time()

    print(f"[{label}] Waiting… (Ctrl+C to exit)")
    # device=None -> uses sd.default.device[0] that we force-bound
    with sd.InputStream(samplerate=device_sr, channels=1, dtype="int16",
                        blocksize=fdev, device=None) as stream:
        while True:
            if not in_speech and (time.time() - start) > max_wait_s:
                print(f"[{label}] no speech timeout"); return ""
            data,_ = stream.read(fdev)
            mono = data[:,0]
            f16  = _to_16k(mono, device_sr)
            talking = (_rms_int16(f16) >= ENERGY_GATE) and vad.is_speech(f16.tobytes(), 16000)

            if not in_speech:
                ring.append(mono)
                if len(ring) > pre_f: ring.pop(0)
                if talking:
                    in_speech = True
                    collected.extend(ring); ring.clear()
                    collected.append(mono)
                    total_frames=1; speech_frames=1; sil_cnt=0
                    print(f"[{label}] speech start")
            else:
                collected.append(mono); total_frames += 1
                if talking: speech_frames += 1; sil_cnt = 0
                else: sil_cnt += 1
                if sil_cnt >= sil_f or (total_frames*FRAME_MS/1000.0) >= max_s:
                    break

    if not collected or speech_frames < min_f:
        print(f"[{label}] too short/garbled (speech_frames={speech_frames}), ignoring.")
        return ""

    y16 = _to_16k(np.concatenate(collected, axis=0), device_sr)
    sf.write(path16, y16, 16000, subtype="PCM_16")
    print(f"[{label}] captured {len(y16)/16000.0:.2f}s")
    time.sleep(0.05)
    return path16
