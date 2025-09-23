import json, os
STATE = "/tmp/kb_motion_state.json"

def _load():
    try:
        with open(STATE,"r") as f: return json.load(f)
    except: return {"n": 0}

def _save(s):
    try:
        with open(STATE,"w") as f: json.dump(s,f)
    except: pass

def next_moves():
    s = _load()
    n = int(s.get("n", 0))
    # Alternate CW / CCW
    core = ["SPIN 260"] if (n % 2) == 0 else ["SPIN_CCW 260"]
    # Every 5th move: brief recenter wiggle instead of full spin
    if (n % 5) == 4:
        core = ["SPIN_CCW 140", "SPIN 140"]
    # LED on for visibility during motion; OFF after
    moves = ["EMO:PLAY", "LED ON"] + core + ["LED OFF"]
    s["n"] = n + 1
    _save(s)
    return moves
