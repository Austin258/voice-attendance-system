import os
import numpy as np
import librosa
import pickle
from scipy.stats import kurtosis, skew

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000
N_MFCC          = 40
THRESHOLD       = 0.75      # raised to avoid false positives on real recordings
OUTPUT_DIR      = "outputs/features/anti_spoofing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION FOR ANTI-SPOOFING
# ─────────────────────────────────────────────
def extract_spoofing_features(audio, sr):
    """
    Extract acoustic features that distinguish live vs replayed audio.
    Replayed audio tends to have:
      - Lower spectral entropy (flatter spectrum)
      - Higher kurtosis (more peaky distribution)
      - Lower zero crossing rate variation
      - Compressed dynamic range
      - Unnatural MFCC delta patterns
    """
    features = {}

    # 1. MFCCs and their deltas
    mfcc        = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features["mfcc_mean"]         = np.mean(mfcc, axis=1)
    features["mfcc_std"]          = np.std(mfcc, axis=1)
    features["mfcc_delta_mean"]   = np.mean(mfcc_delta, axis=1)
    features["mfcc_delta2_mean"]  = np.mean(mfcc_delta2, axis=1)

    # 2. Spectral features
    spec_centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spec_flatness  = librosa.feature.spectral_flatness(y=audio)[0]

    features["spec_centroid_mean"]  = np.mean(spec_centroid)
    features["spec_centroid_std"]   = np.std(spec_centroid)
    features["spec_bandwidth_mean"] = np.mean(spec_bandwidth)
    features["spec_rolloff_mean"]   = np.mean(spec_rolloff)
    features["spec_flatness_mean"]  = np.mean(spec_flatness)
    features["spec_flatness_std"]   = np.std(spec_flatness)

    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"]  = np.std(zcr)

    # 4. RMS Energy
    rms = librosa.feature.rms(y=audio)[0]
    features["rms_mean"]  = np.mean(rms)
    features["rms_std"]   = np.std(rms)
    features["rms_range"] = np.max(rms) - np.min(rms)

    # 5. Statistical properties
    features["signal_kurtosis"] = kurtosis(audio)
    features["signal_skew"]     = skew(audio)

    # 6. Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"]  = np.std(chroma)

    # 7. Fundamental frequency (F0)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([0])
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

    features["f0_mean"] = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
    features["f0_std"]  = np.std(f0_voiced)  if len(f0_voiced) > 0 else 0

    return features

def flatten_features(features):
    """Flatten all feature values into a single 1D vector."""
    vector = []
    for key in sorted(features.keys()):
        val = features[key]
        if isinstance(val, np.ndarray):
            vector.extend(val.tolist())
        else:
            vector.append(float(val))
    return np.array(vector)

# ─────────────────────────────────────────────
# RULE-BASED SPOOF SCORING
# ─────────────────────────────────────────────
def compute_spoof_score(features):
    """
    Compute a spoof likelihood score between 0 and 1.
    Score > THRESHOLD → SPOOFED
    Score < THRESHOLD → LIVE

    Threshold raised to 0.75 to prevent false positives
    on genuine student recordings collected in varied
    acoustic environments.

    Rules based on known acoustic differences:
    - High spectral flatness     → replay indicator
    - Low ZCR std                → unnatural = replay
    - Low RMS std                → compressed dynamic range = replay
    - High kurtosis              → peaky signal = replay
    - Low F0 std                 → unnatural pitch = replay
    """
    score   = 0.0
    reasons = []

    # Rule 1: Spectral flatness
    if features["spec_flatness_mean"] > 0.02:
        score += 0.25
        reasons.append("High spectral flatness (replay indicator)")

    # Rule 2: ZCR std
    if features["zcr_std"] < 0.02:
        score += 0.20
        reasons.append("Low ZCR variation (unnatural speech pattern)")

    # Rule 3: RMS std
    if features["rms_std"] < 0.01:
        score += 0.20
        reasons.append("Low RMS variation (compressed dynamic range)")

    # Rule 4: Signal kurtosis
    if features["signal_kurtosis"] > 10:
        score += 0.20
        reasons.append("High signal kurtosis (peaky signal distribution)")

    # Rule 5: F0 std
    if features["f0_std"] < 5.0:
        score += 0.15
        reasons.append("Low F0 variation (unnatural pitch = possible replay)")

    return round(min(score, 1.0), 4), reasons

# ─────────────────────────────────────────────
# MAIN ANTI-SPOOFING FUNCTION
# ─────────────────────────────────────────────
def check_liveness(audio, sr, student_id="unknown", verbose=True):
    """
    Full anti-spoofing pipeline for a single audio input.
    Returns: dict with verdict, score, and reasons
    """
    features       = extract_spoofing_features(audio, sr)
    score, reasons = compute_spoof_score(features)
    verdict        = "SPOOFED" if score >= THRESHOLD else "LIVE"

    result = {
        "student_id" : student_id,
        "spoof_score": score,
        "verdict"    : verdict,
        "reasons"    : reasons,
        "threshold"  : THRESHOLD
    }

    if verbose:
        status_icon = "🔴 SPOOFED" if verdict == "SPOOFED" else "🟢 LIVE"
        print(f"\n  Student   : {student_id}")
        print(f"  Verdict   : {status_icon}")
        print(f"  Score     : {score} / 1.0  (threshold: {THRESHOLD})")
        if reasons:
            print(f"  Reasons   :")
            for r in reasons:
                print(f"    - {r}")

    return result

# ─────────────────────────────────────────────
# BATCH VALIDATION ON EXISTING VOICE SAMPLES
# ─────────────────────────────────────────────
def validate_all_samples(voice_dir="data/voice_samples",
                          n_students=32, n_recordings=7):
    """
    Run anti-spoofing check on all collected voice samples.
    Since these are genuine recordings, most should pass as LIVE.
    """
    print("=" * 55)
    print("   ANTI-SPOOFING BATCH VALIDATION")
    print("=" * 55)

    results  = []
    spoofed  = []
    live     = []
    errors   = []

    for i in range(1, n_students + 1):
        student_id = f"student{i}"
        for j in range(1, n_recordings + 1):
            filepath = os.path.join(
                voice_dir, student_id, f"{student_id}_{j}.wav"
            )
            if not os.path.exists(filepath):
                errors.append(filepath)
                continue
            try:
                audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                result    = check_liveness(
                    audio, sr,
                    student_id=f"{student_id}_rec{j}",
                    verbose=False
                )
                results.append(result)
                if result["verdict"] == "SPOOFED":
                    spoofed.append(result)
                else:
                    live.append(result)
            except Exception as e:
                errors.append(f"{filepath}: {str(e)}")

    # Save results
    import json
    out_path = os.path.join(OUTPUT_DIR, "batch_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n  Total checked : {len(results)}")
    print(f"  🟢 LIVE       : {len(live)}")
    print(f"  🔴 SPOOFED    : {len(spoofed)}")
    print(f"  ⚠️  Errors     : {len(errors)}")

    if spoofed:
        print(f"\n  ⚠️  Flagged files:")
        for s in spoofed:
            print(f"    - {s['student_id']} | score: {s['spoof_score']}")
            for r in s["reasons"]:
                print(f"      → {r}")
    else:
        print(f"\n  ✅ All recordings passed liveness check")

    print(f"\n  ✅ Results saved to: {out_path}")
    return results

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    validate_all_samples()

    print("\n" + "=" * 55)
    print("✅ Anti-Spoofing module complete!")
    print("🚀 Ready for Stage 4: Data Splitting")
    print("=" * 55)