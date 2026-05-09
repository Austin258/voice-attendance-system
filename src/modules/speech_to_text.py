import os
import re
import numpy as np
import soundfile as sf
import speech_recognition as sr
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TEMP_DIR = "outputs/temp_recordings"
os.makedirs(TEMP_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SPEECH TO TEXT MODULE
# ─────────────────────────────────────────────
class SpeechToTextModule:
    """
    Converts recorded audio feedback to text using
    Google Speech Recognition API.
    Used specifically for the sentiment feedback component.
    Falls back gracefully when internet is unavailable.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold         = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold          = 0.8

    # ─────────────────────────────────────────
    # AUDIO FILE → TEXT
    # ─────────────────────────────────────────
    def audio_file_to_text(self, filepath):
        """
        Convert a .wav audio file to text.
        Returns (text, success) tuple.
        """
        print(f"\n  🔄 Converting speech to text...")

        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}")
            return "", False

        try:
            with sr.AudioFile(filepath) as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=0.5)
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(
                audio_data, language="en-US")

            print(f"  ✅ Transcription successful")
            print(f"  📝 Text: \"{text}\"")
            return text, True

        except sr.UnknownValueError:
            print(f"  ⚠️  Could not understand audio clearly")
            print(f"  💡 Tip: Speak clearly, reduce background noise")
            return "", False
        except sr.RequestError as e:
            print(f"  ❌ Speech recognition service unavailable: {e}")
            print(f"  💡 Check your internet connection")
            return "", False
        except Exception as e:
            print(f"  ❌ Transcription error: {e}")
            return "", False

    # ─────────────────────────────────────────
    # NUMPY ARRAY → TEXT
    # ─────────────────────────────────────────
    def audio_array_to_text(self, audio, sample_rate=16000):
        """Convert numpy audio array to text via temp file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(TEMP_DIR, f"stt_{timestamp}.wav")
        try:
            sf.write(temp_path, audio, sample_rate)
            return self.audio_file_to_text(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ─────────────────────────────────────────
    # PREPROCESS TEXT
    # ─────────────────────────────────────────
    def preprocess_text(self, text):
        """Clean and normalize transcribed text."""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ─────────────────────────────────────────
    # FULL PIPELINE: AUDIO FILE → CLEAN TEXT
    # ─────────────────────────────────────────
    def process_feedback_audio(self, filepath):
        """
        Complete pipeline: audio file → cleaned text.
        Returns (clean_text, success) tuple.
        """
        raw_text, success = self.audio_file_to_text(filepath)

        if not success or not raw_text:
            return "", False

        clean_text = self.preprocess_text(raw_text)
        print(f"  ✅ Cleaned: \"{clean_text}\"")
        return clean_text, True


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   SPEECH TO TEXT MODULE TEST")
    print("=" * 55)

    stt       = SpeechToTextModule()
    test_file = "data/voice_samples/student1/student1_1.wav"

    if os.path.exists(test_file):
        text, success = stt.audio_file_to_text(test_file)
        if success:
            clean = stt.preprocess_text(text)
            print(f"\n✅ Speech-to-text working!")
            print(f"   Raw   : {text}")
            print(f"   Clean : {clean}")
        else:
            print("\n⚠️  Transcription failed (check internet)")
    else:
        print("⚠️  Test file not found.")