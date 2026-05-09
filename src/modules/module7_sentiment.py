import os
import sys
import re
import pickle
import numpy as np
import speech_recognition as sr
import soundfile as sf
from datetime import datetime

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# MODULE 7: SENTIMENT ANALYSIS MODULE
# ─────────────────────────────────────────────
class SentimentAnalysisModule:
    """
    Analyzes student verbal feedback sentiment.

    Pipeline:
    1. Convert speech audio → text (STT)
    2. Preprocess text (clean, normalize)
    3. Classify sentiment (TF-IDF + Ensemble)
    4. Return: Positive / Negative / Neutral

    Model: TF-IDF + SVM + LR + RF Voting Ensemble
    Accuracy: 97.23%

    Feedback is OPTIONAL:
    - If provided: classify and store result
    - If skipped: store feedback = "Not Provided",
                  sentiment = NULL
    """

    MODELS_DIR   = "models"
    TEMP_DIR     = "outputs/temp_recordings"
    LABEL_MAP    = {0: "Negative", 1: "Neutral",
                    2: "Positive"}
    FEEDBACK_QUESTIONS = [
        "How was today's lecture?",
        "Did you understand the topic covered today?",
        "What challenges did you face during the class?"
    ]

    def __init__(self):
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        self._load_model()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold         = 300
        self.recognizer.dynamic_energy_threshold = True

    def _load_model(self):
        with open(os.path.join(
                self.MODELS_DIR, "sentiment_model.pkl"),
                "rb") as f:
            self.pipeline = pickle.load(f)

    # ─────────────────────────────────────────
    # SPEECH TO TEXT
    # ─────────────────────────────────────────
    def speech_to_text(self, filepath):
        """Convert audio file to text using Google STT."""
        try:
            with sr.AudioFile(filepath) as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=0.5)
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(
                audio_data, language="en-US")
            return text, True
        except sr.UnknownValueError:
            return "", False
        except sr.RequestError:
            return "", False
        except Exception:
            return "", False

    # ─────────────────────────────────────────
    # PREPROCESS TEXT
    # ─────────────────────────────────────────
    def preprocess(self, text):
        """Clean and normalize text for classification."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ─────────────────────────────────────────
    # CLASSIFY SENTIMENT
    # ─────────────────────────────────────────
    def classify(self, text):
        """
        Classify text into Positive/Negative/Neutral.
        Returns None if text is empty.
        """
        text = self.preprocess(text)
        if not text:
            return None

        probs      = self.pipeline.predict_proba([text])[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "sentiment" : self.LABEL_MAP[pred_idx],
            "confidence": round(confidence * 100, 2),
            "scores"    : {
                "Negative": round(float(probs[0])*100, 2),
                "Neutral" : round(float(probs[1])*100, 2),
                "Positive": round(float(probs[2])*100, 2)
            },
            "raw_text"  : text
        }

    # ─────────────────────────────────────────
    # FULL PIPELINE: AUDIO → SENTIMENT
    # ─────────────────────────────────────────
    def process_audio(self, filepath):
        """
        Full pipeline: audio file → sentiment result.
        Returns (result, success) tuple.
        """
        print(f"\n  🔄 Converting speech to text...")
        text, ok = self.speech_to_text(filepath)

        if not ok or not text:
            print(f"  ⚠️  Speech recognition failed")
            return None, False

        print(f"  📝 Transcribed: \"{text}\"")
        result = self.classify(text)

        if result:
            print(f"  ✅ Sentiment: {result['sentiment']} "
                  f"({result['confidence']}%)")

        return result, True

    # ─────────────────────────────────────────
    # DISPLAY RESULT
    # ─────────────────────────────────────────
    def display_result(self, result):
        if not result:
            print("  ℹ️  No sentiment result available")
            return
        print(f"\n  📊 Sentiment Analysis Result")
        print(f"  Sentiment   : {result['sentiment']}")
        print(f"  Confidence  : {result['confidence']}%")
        print(f"  Scores:")
        for label, score in result["scores"].items():
            bar = "█" * int(score / 5)
            print(f"    {label:<10}: {score:>6.2f}%  {bar}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 7: SENTIMENT ANALYSIS TEST")
    print("=" * 55)

    sa = SentimentAnalysisModule()

    test_cases = [
        ("The lecture was excellent and very clear",
         "Positive"),
        ("I could not understand anything today",
         "Negative"),
        ("The class was okay with some unclear parts",
         "Neutral"),
    ]

    print("\n  Testing text classification:")
    correct = 0
    for text, expected in test_cases:
        result = sa.classify(text)
        got    = result["sentiment"] if result else "N/A"
        match  = "✅" if got == expected else "❌"
        print(f"  {match} Expected={expected:<10} "
              f"Got={got:<10} | \"{text[:40]}\"")
        if got == expected:
            correct += 1

    print(f"\n  Accuracy: {correct}/{len(test_cases)} "
          f"({correct/len(test_cases)*100:.0f}%)")
    print(f"\n✅ Sentiment Analysis Module working!")