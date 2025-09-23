import random, threading, time

class MotionPlanner:
    def __init__(self, home_every=(5,10)):
        self._last_side = None
        self._moves = 0
        self._home_every = home_every

    def next_move(self):
        "Return a tuple ('L'|'R'|'HOME', duration_ms, speed_0_255)"
        self._moves += 1
        # Home bias: after X moves, return to home
        if self._moves >= random.randint(*self._home_every):
            self._moves = 0
            return ("HOME", 600, 180)
        # Alternate L/R
        side = 'L' if self._last_side != 'L' else 'R'
        self._last_side = side
        return (side, random.randint(350,650), random.randint(140,200))

QUIPS = [
    "I couldn’t hear you.",
    "My ears are buffering.",
    "Audio was fuzzy—try again?",
]

JOKES = [
    "I told my computer I needed more space. It uninstalled Windows.",
    "I started a side business selling invisible ink. Profits are through the roof—just not visible yet.",
    "Why do programmers hate nature? It has too many bugs.",
]

def no_speech_fallback():
    return f"{random.choice(QUIPS)} Here’s a joke: {random.choice(JOKES)}"

class SpeechGate:
    """Very lightweight 'interrupt while talking'.
       Your TTS must call `gate.check()` between chunks and stop if returns False."""
    def __init__(self):
        self._lock = threading.Lock()
        self._speaking = False
        self._allow = True

    def begin(self):
        with self._lock:
            self._speaking = True
            self._allow = True

    def stop(self):
        with self._lock:
            self._allow = False
            self._speaking = False

    def check(self):
        with self._lock:
            return self._allow

    def is_speaking(self):
        with self._lock:
            return self._speaking

def should_interrupt(transcribed_text: str) -> bool:
    return 'kinderbot' in (transcribed_text or '').lower()

# ---- Integration helpers (wire these to your serial/motor layer) ----
def send_motion(serial_send_fn, move_tuple):
    which, ms, spd = move_tuple
    if which == "HOME":
        serial_send_fn("HOME\n")
        time.sleep(ms/1000)
        serial_send_fn("STOP\n")
        return
    if which == "L":
        # Turn left by driving RIGHT wheel forward (example)
        serial_send_fn(f"RIGHT {spd}\n"); time.sleep(ms/1000); serial_send_fn("STOP\n")
    else:
        # Turn right by driving LEFT wheel forward (example)
        serial_send_fn(f"LEFT {spd}\n");  time.sleep(ms/1000); serial_send_fn("STOP\n")
