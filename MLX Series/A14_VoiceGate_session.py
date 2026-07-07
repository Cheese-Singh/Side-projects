import sys
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from A14_VoiceGate import VoiceGate

SAMPLE_RATE = 16000
gate = VoiceGate(voiceprint_path="Gwen_voiceprint.pt", threshold=0.30)

# ---------------------
# RECORDING MICROPHONE
# ---------------------

def record_clip(path, seconds=5):
    print(f"Recording {seconds}s, speak now...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, SAMPLE_RATE)
    print(f"Saved {path}")

# --------------------
# ONE-TIME ENROLLMENT
# --------------------

def run_enrollment(enrollment_dir=None, live_count=30):
    if enrollment_dir:
        wav_paths = sorted(Path(enrollment_dir).glob("*.wav"))
    else:
        tmp_dir = Path("enrollment_clips")
        tmp_dir.mkdir(exist_ok=True)
        wav_paths = []
        for i in range(live_count):
            input(f"Press enter, then speak naturally for clip {i + 1}/{live_count}...")
            path = tmp_dir / f"enroll_{i}.wav"
            record_clip(path, seconds=4)
            wav_paths.append(path)

    if len(wav_paths) < 5:
        print("Record at least 5-10 short clips (different times/rooms) for a stable voiceprint.")
        return

    gate.enroll(wav_paths)
    print(f"Enrolled from {len(wav_paths)} clips -> {gate.voiceprint_path}")

# ----------------------
# THRESHOLD CALIBRATION
# ----------------------

def check_score(label="test"):
    path = Path(f"{label}.wav")
    record_clip(path, seconds=4)
    score, passed = gate.verify_file(path)
    print(f"score={score:.3f} passed={passed} (threshold={gate.threshold})")
    return score

# -------------
# STARTUP GATE
# -------------

def unlock_session():
    path = Path("startup.wav")
    record_clip(path, seconds=4)
    score, passed = gate.verify_file(path)
    print(f"Startup verification score={score:.3f} passed={passed}")
    return passed

# ----------------------
# SENSITIVE-ACTION GATE
# ----------------------

def guarded_action(action_name, action_fn):
    print(f"'{action_name}' requires re-verification.")
    path = Path("confirm.wav")
    record_clip(path, seconds=3)
    score, passed = gate.verify_file(path)
    print(f"Re-verification for '{action_name}' score={score:.3f} passed={passed}")
    if not passed:
        print(f"Blocked: '{action_name}' requires your verified voice.")
        return None
    return action_fn()

# ------------------
# MOCK GWEN SESSION
# ------------------

def delete_file_stub():
    print("Deleting file...")
    return True

def read_memory_stub():
    print("Reading user memory...")
    return True

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "session"

    if mode == "enroll":
        run_enrollment(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "score":
        check_score(sys.argv[2] if len(sys.argv) > 2 else "test")
    else:
        if not unlock_session():
            print("Session not started: voice not recognized.")
            sys.exit(1)

        print("Gwen session unlocked, listening normally...")
        guarded_action("delete_file", delete_file_stub)
        guarded_action("read_memory", read_memory_stub)