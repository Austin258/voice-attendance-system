import os
import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append(os.path.abspath("."))
from src.modules.module3_preprocessing import PreprocessingModule

# ─────────────────────────────────────────────
# MODULE 5: VOICE VERIFICATION MODULE
# ─────────────────────────────────────────────
class VoiceVerificationModule:
    """
    Verifies student identity using voice biometrics.

    Addresses Impersonation/Proxy Attendance by:
    - CNN-LSTM model for speaker verification
    - Test-Time Augmentation (TTA) for robustness
    - Confidence threshold enforcement
    - Top-3 prediction transparency

    Architecture: CNN (spatial) + LSTM (temporal)
    Accuracy: 97.43%
    """

    MODELS_DIR           = "models"
    CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self.preproc = PreprocessingModule()
        self._load_model()

    def _load_model(self):
        self.model = tf.keras.models.load_model(
            os.path.join(self.MODELS_DIR,
                         "voice_recognition_model.keras")
        )
        with open(os.path.join(
                self.MODELS_DIR,
                "../outputs/features/mfcc/label_encoder.pkl"),
                "rb") as f:
            self.label_encoder = pickle.load(f)

    # ─────────────────────────────────────────
    # PREDICT WITH TTA
    # ─────────────────────────────────────────
    def predict(self, audio, sr, n_tta=5):
        """
        Predict student identity from audio using
        Test-Time Augmentation for robustness.
        Averages predictions over n_tta passes.
        """
        mfcc      = self.preproc.extract_mfcc(audio, sr)
        mfcc_base = mfcc[np.newaxis, ..., np.newaxis]

        all_probs = [self.model.predict(
            mfcc_base, verbose=0)]

        for _ in range(n_tta - 1):
            noise = np.random.normal(0, 0.002,
                                      mfcc_base.shape)
            all_probs.append(self.model.predict(
                mfcc_base + noise, verbose=0))

        avg_probs  = np.mean(all_probs, axis=0)[0]
        pred_idx   = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        student_id = self.label_encoder.inverse_transform(
            [pred_idx])[0]

        top3 = [
            {
                "student"   : self.label_encoder
                              .inverse_transform([i])[0],
                "confidence": round(
                    float(avg_probs[i]) * 100, 2)
            }
            for i in np.argsort(avg_probs)[::-1][:3]
        ]

        return {
            "student_id": student_id,
            "confidence": round(confidence * 100, 2),
            "verified"  : confidence >= self.CONFIDENCE_THRESHOLD,
            "top3"      : top3,
            "threshold" : self.CONFIDENCE_THRESHOLD * 100
        }

    # ─────────────────────────────────────────
    # VERIFY FROM FILE
    # ─────────────────────────────────────────
    def verify_from_file(self, filepath):
        """Load, preprocess and verify from audio file."""
        audio, sr, _ = self.preproc.process(filepath)
        return self.predict(audio, sr)

    # ─────────────────────────────────────────
    # DISPLAY RESULT
    # ─────────────────────────────────────────
    def display_result(self, result):
        print(f"\n  🔊 Voice Verification Result")
        print(f"  Predicted   : {result['student_id']}")
        print(f"  Confidence  : {result['confidence']}%")
        print(f"  Verified    : "
              f"{'✅ Yes' if result['verified'] else '❌ No'}")
        print(f"  Threshold   : {result['threshold']}%")
        print(f"  Top 3:")
        for r, p in enumerate(result["top3"], 1):
            print(f"    {r}. {p['student']:<12} "
                  f"→ {p['confidence']}%")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 5: VOICE VERIFICATION TEST")
    print("=" * 55)

    vv        = VoiceVerificationModule()
    test_file = "data/voice_samples/student2/student2_1.wav"

    if os.path.exists(test_file):
        result = vv.verify_from_file(test_file)
        vv.display_result(result)
        print(f"\n✅ Voice Verification Module working!")
    else:
        print("⚠️  Test file not found")