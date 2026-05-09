import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE         = 16000
CHANNELS            = 1
ATTENDANCE_DURATION = 5      # seconds for attendance phrase
FEEDBACK_MIN        = 5      # minimum feedback duration
FEEDBACK_MAX        = 30     # maximum feedback duration
TEMP_DIR            = "outputs/temp_recordings"
ATTENDANCE_PHRASE   = "My name is [Your Name], I am present"

os.makedirs(TEMP_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# VOICE CAPTURE MODULE
# ─────────────────────────────────────────────
class VoiceCaptureModule:
    """
    Captures real-time audio input from system microphone.
    Handles both attendance phrase recording and
    optional feedback recording (5–30 seconds).
    """

    def __init__(self, sample_rate=SAMPLE_RATE,
                 channels=CHANNELS):
        self.sample_rate = sample_rate
        self.channels    = channels

    # ─────────────────────────────────────────
    # CHECK MICROPHONE
    # ─────────────────────────────────────────
    def check_microphone(self):
        """Verify microphone is available."""
        try:
            devices       = sd.query_devices()
            input_devices = [
                d for d in devices
                if d['max_input_channels'] > 0
            ]
            if not input_devices:
                print("  ❌ No microphone found.")
                return False
            device_name = sd.query_devices(kind='input')['name']
            print(f"  ✅ Microphone detected: {device_name}")
            return True
        except Exception as e:
            print(f"  ❌ Microphone check failed: {e}")
            return False

    # ─────────────────────────────────────────
    # RECORD AUDIO (CORE)
    # ─────────────────────────────────────────
    def _record(self, duration, label="Recording"):
        """Core audio recording function."""
        print(f"\n  ⏳ {label} starts in:")
        for i in range(3, 0, -1):
            print(f"     {i}...", end="\r")
            time.sleep(1)

        print(f"  🔴 RECORDING... ({duration} seconds)")
        print(f"  {'─'*40}")

        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()

    # ─────────────────────────────────────────
    # AUDIO QUALITY CHECK
    # ─────────────────────────────────────────
    def _check_quality(self, audio):
        """Check audio for silence or clipping."""
        if len(audio) == 0:
            return False, "Empty recording"

        rms     = np.sqrt(np.mean(audio ** 2))
        max_amp = np.max(np.abs(audio))

        if rms < 0.001:
            return False, "Audio too quiet — speak louder"
        if max_amp > 0.98:
            return False, "Audio clipped — move mic further away"

        return True, f"OK (RMS: {rms:.4f})"

    # ─────────────────────────────────────────
    # SAVE AUDIO
    # ─────────────────────────────────────────
    def _save_audio(self, audio, student_id, purpose):
        """Save audio to temp directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{student_id}_{purpose}_{timestamp}.wav"
        filepath  = os.path.join(TEMP_DIR, filename)
        sf.write(filepath, audio, self.sample_rate)
        return filepath

    # ─────────────────────────────────────────
    # RECORD ATTENDANCE PHRASE
    # ─────────────────────────────────────────
    def record_attendance(self, student_id="unknown",
                           max_attempts=3):
        """
        Record student attendance phrase.
        Duration: fixed 5 seconds.
        Phrase: 'My name is [Name], I am present'
        """
        print("\n" + "─" * 45)
        print("  🎙️  VOICE CAPTURE — ATTENDANCE")
        print("─" * 45)
        print(f"\n  📢 Please say the following phrase clearly:")
        print(f'     "{ATTENDANCE_PHRASE}"')

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"\n  🔁 Attempt {attempt} of {max_attempts}")

            try:
                audio = self._record(
                    ATTENDANCE_DURATION,
                    label="Attendance recording"
                )
                passed, msg = self._check_quality(audio)

                if passed:
                    filepath = self._save_audio(
                        audio, student_id, "attendance")
                    print(f"  ✅ Attendance audio captured")
                    return audio, self.sample_rate, filepath
                else:
                    print(f"  ⚠️  Quality issue: {msg}")
                    if attempt < max_attempts:
                        print(f"  🔁 Please try again")

            except Exception as e:
                print(f"  ❌ Recording error: {e}")

        print(f"  ❌ Max attempts reached.")
        return None, None, None

    # ─────────────────────────────────────────
    # RECORD FEEDBACK (OPTIONAL)
    # ─────────────────────────────────────────
    def record_feedback(self, student_id="unknown",
                         duration=15, max_attempts=2):
        """
        Record optional verbal feedback.
        Duration: 5–30 seconds (default 15s).
        Student chooses to provide or skip.
        """
        # Clamp duration within allowed range
        duration = max(FEEDBACK_MIN, min(duration, FEEDBACK_MAX))

        print("\n" + "─" * 45)
        print("  🎙️  VOICE CAPTURE — FEEDBACK")
        print("─" * 45)
        print(f"\n  📢 Please share your feedback on today's lecture")
        print(f"     You have {duration} seconds to speak")
        print(f"     Talk about: what you understood, what was")
        print(f"     unclear, or how you felt about the class")

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"\n  🔁 Attempt {attempt} of {max_attempts}")

            try:
                audio = self._record(
                    duration, label="Feedback recording")
                passed, msg = self._check_quality(audio)

                if passed:
                    filepath = self._save_audio(
                        audio, student_id, "feedback")
                    print(f"  ✅ Feedback audio captured")
                    return audio, self.sample_rate, filepath
                else:
                    print(f"  ⚠️  Quality issue: {msg}")
                    if attempt < max_attempts:
                        print(f"  🔁 Please try again")

            except Exception as e:
                print(f"  ❌ Recording error: {e}")

        print(f"  ❌ Could not capture feedback audio.")
        return None, None, None

    # ─────────────────────────────────────────
    # RECORD WITH RETRY (GENERIC)
    # ─────────────────────────────────────────
    def record_with_retry(self, student_id="unknown",
                           purpose="attendance",
                           max_attempts=3):
        """Generic record with retry for backward compatibility."""
        if purpose == "attendance":
            return self.record_attendance(
                student_id=student_id,
                max_attempts=max_attempts
            )
        else:
            return self.record_feedback(
                student_id=student_id,
                max_attempts=max_attempts
            )

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────
    def cleanup_temp(self, filepath):
        """Remove temporary audio file."""
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   VOICE CAPTURE MODULE TEST")
    print("=" * 55)

    capture = VoiceCaptureModule()

    if capture.check_microphone():
        print("\n[1] Testing attendance recording...")
        audio, sr, fp = capture.record_attendance(
            student_id="test_student")
        if audio is not None:
            print(f"✅ Attendance capture passed | File: {fp}")
            capture.cleanup_temp(fp)

        print("\n[2] Testing feedback recording (15s)...")
        audio, sr, fp = capture.record_feedback(
            student_id="test_student", duration=15)
        if audio is not None:
            print(f"✅ Feedback capture passed | File: {fp}")
            capture.cleanup_temp(fp)