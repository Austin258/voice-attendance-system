import os
import sys
import numpy as np
import librosa
from scipy.stats import kurtosis, skew

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# MODULE 6: ANTI-SPOOFING MODULE
# ─────────────────────────────────────────────
class AntiSpoofingModule:
    """
    Detects fraudulent voice inputs.

    Addresses Security Weakness by:
    - Detecting replay attacks (pre-recorded audio)
    - Analyzing acoustic anomalies
    - Scoring spoof likelihood 0–1
    - Blocking any score >= threshold (0.75)

    Features analyzed:
    - Spectral flatness (replay = flat spectrum)
    - Zero crossing rate variation
    - RMS energy dynamic range
    - Signal kurtosis
    - F0 pitch variation
    """

    THRESHOLD   = 0.75
    SAMPLE_RATE = 16000

    # ─────────────────────────────────────────
    # EXTRACT FEATURES
    # ─────────────────────────────────────────
    def extract_features(self, audio, sr):
        """Extract acoustic features for spoof detection."""
        features = {}

        spec_flatness = librosa.feature.spectral_flatness(
            y=audio)[0]
        features["spec_flatness_mean"] = np.mean(spec_flatness)

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features["zcr_std"] = np.std(zcr)

        rms = librosa.feature.rms(y=audio)[0]
        features["rms_std"] = np.std(rms)

        features["signal_kurtosis"] = kurtosis(audio)

        try:
            f0, voiced, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            f0v = f0[voiced] if voiced is not None \
                  else np.array([0])
            f0v = f0v[~np.isnan(f0v)]
            features["f0_std"] = (
                np.std(f0v) if len(f0v) > 0 else 0)
        except Exception:
            features["f0_std"] = 0

        return features

    # ─────────────────────────────────────────
    # COMPUTE SPOOF SCORE
    # ─────────────────────────────────────────
    def compute_score(self, features):
        """
        Score spoof likelihood 0.0 – 1.0.
        Score >= 0.75 → SPOOFED
        Score <  0.75 → LIVE
        """
        score   = 0.0
        reasons = []

        if features["spec_flatness_mean"] > 0.02:
            score += 0.25
            reasons.append(
                "High spectral flatness (replay indicator)")

        if features["zcr_std"] < 0.02:
            score += 0.20
            reasons.append(
                "Low ZCR variation (unnatural pattern)")

        if features["rms_std"] < 0.01:
            score += 0.20
            reasons.append(
                "Low RMS variation (compressed range)")

        if features["signal_kurtosis"] > 10:
            score += 0.20
            reasons.append(
                "High kurtosis (peaky distribution)")

        if features["f0_std"] < 5.0:
            score += 0.15
            reasons.append(
                "Low F0 variation (unnatural pitch)")

        return round(min(score, 1.0), 4), reasons

    # ─────────────────────────────────────────
    # CHECK LIVENESS
    # ─────────────────────────────────────────
    def check(self, audio, sr,
               student_id="unknown", verbose=True):
        """
        Full liveness check pipeline.
        Returns result dict with verdict and score.
        """
        features       = self.extract_features(audio, sr)
        score, reasons = self.compute_score(features)
        verdict        = "SPOOFED" if score >= self.THRESHOLD \
                         else "LIVE"

        result = {
            "student_id" : student_id,
            "spoof_score": score,
            "verdict"    : verdict,
            "reasons"    : reasons,
            "threshold"  : self.THRESHOLD
        }

        if verbose:
            icon = "🔴 SPOOFED" if verdict == "SPOOFED" \
                   else "🟢 LIVE"
            print(f"\n  Student   : {student_id}")
            print(f"  Verdict   : {icon}")
            print(f"  Score     : {score} / 1.0 "
                  f"(threshold: {self.THRESHOLD})")
            if reasons:
                print(f"  Indicators:")
                for r in reasons:
                    print(f"    - {r}")

        return result

    # ─────────────────────────────────────────
    # CHECK FROM FILE
    # ─────────────────────────────────────────
    def check_from_file(self, filepath,
                         student_id="unknown",
                         verbose=True):
        audio, sr = librosa.load(filepath,
                                  sr=self.SAMPLE_RATE)
        return self.check(audio, sr, student_id, verbose)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 6: ANTI-SPOOFING TEST")
    print("=" * 55)

    spoof     = AntiSpoofingModule()
    test_file = "data/voice_samples/student1/student1_1.wav"

    if os.path.exists(test_file):
        result = spoof.check_from_file(
            test_file, "student1", verbose=True)
        print(f"\n✅ Anti-Spoofing Module working!")
        print(f"   Verdict: {result['verdict']}")
    else:
        print("⚠️  Test file not found")