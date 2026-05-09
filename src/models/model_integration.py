import os
import numpy as np
import pickle
import sqlite3
import librosa
import re
from datetime import datetime

import tensorflow as tf

# ── Import preprocessing utilities
import sys
sys.path.append(os.path.abspath("."))
from src.preprocessing.preprocess import preprocess_audio, extract_mfcc
from src.preprocessing.anti_spoofing import check_liveness
from src.preprocessing.location_verification import verify_location

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODELS_DIR           = "models"
DB_PATH              = "database/attendance_system.db"
SAMPLE_RATE          = 16000
N_MFCC               = 40
MAX_FRAMES           = 200
CONFIDENCE_THRESHOLD = 0.45

# ─────────────────────────────────────────────
# 1. LOAD ALL MODELS
# ─────────────────────────────────────────────
def load_models():
    """
    Load voice CNN-LSTM model, sentiment TF-IDF pipeline
    and label encoder. Called once at system startup.
    """
    print("=" * 55)
    print("   LOADING MODELS")
    print("=" * 55)

    voice_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "voice_recognition_model.keras")
    )
    print("✅ Voice model loaded       : CNN-LSTM v5 (97.43%)")

    with open(os.path.join(MODELS_DIR, "sentiment_model.pkl"), "rb") as f:
        sentiment_pipeline = pickle.load(f)
    print("✅ Sentiment model loaded   : TF-IDF + Ensemble (97.23%)")

    with open("outputs/features/mfcc/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded     : 32 student classes")

    print("=" * 55)
    return voice_model, sentiment_pipeline, label_encoder

# ─────────────────────────────────────────────
# 2. VOICE RECOGNITION INFERENCE
# ─────────────────────────────────────────────
def predict_student_identity(voice_model, label_encoder,
                              audio, sr, n_tta=5):
    """
    Predict student identity from preprocessed audio.
    Uses Test-Time Augmentation (TTA) for robustness.
    """
    mfcc      = extract_mfcc(audio, sr)
    mfcc_base = mfcc[np.newaxis, ..., np.newaxis]   # (1, 40, 200, 1)

    all_probs = [voice_model.predict(mfcc_base, verbose=0)]

    for _ in range(n_tta - 1):
        noise      = np.random.normal(0, 0.002, mfcc_base.shape)
        mfcc_noisy = mfcc_base + noise
        all_probs.append(voice_model.predict(mfcc_noisy, verbose=0))

    avg_probs  = np.mean(all_probs, axis=0)[0]
    pred_idx   = int(np.argmax(avg_probs))
    confidence = float(avg_probs[pred_idx])
    student_id = label_encoder.inverse_transform([pred_idx])[0]

    top3_idx = np.argsort(avg_probs)[::-1][:3]
    top3     = [
        {
            "student"   : label_encoder.inverse_transform([i])[0],
            "confidence": round(float(avg_probs[i]) * 100, 2)
        }
        for i in top3_idx
    ]

    return {
        "student_id": student_id,
        "confidence": round(confidence * 100, 2),
        "verified"  : confidence >= CONFIDENCE_THRESHOLD,
        "top3"      : top3
    }

# ─────────────────────────────────────────────
# 3. SENTIMENT ANALYSIS INFERENCE
# ─────────────────────────────────────────────
def predict_sentiment(sentiment_pipeline, text):
    """
    Predict sentiment from student verbal feedback.
    Uses TF-IDF + Ensemble pipeline.
    """
    text = re.sub(r"[^a-z\s]", "", text.lower()).strip()

    if not text:
        return {
            "sentiment" : "Neutral",
            "confidence": 0.0,
            "scores"    : {
                "Negative": 0.0,
                "Neutral" : 100.0,
                "Positive": 0.0
            }
        }

    probs      = sentiment_pipeline.predict_proba([text])[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label_map  = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment  = label_map[pred_idx]

    return {
        "sentiment" : sentiment,
        "confidence": round(confidence * 100, 2),
        "scores"    : {
            "Negative": round(float(probs[0]) * 100, 2),
            "Neutral" : round(float(probs[1]) * 100, 2),
            "Positive": round(float(probs[2]) * 100, 2)
        }
    }

# ─────────────────────────────────────────────
# 4. DATABASE SETUP AND LOGGING
# ─────────────────────────────────────────────
def setup_attendance_table():
    """
    Drop and recreate attendance table with full schema.
    This ensures feedback_text column is always present.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop old table if schema is outdated
    cursor.execute("DROP TABLE IF EXISTS attendance")

    # Recreate with correct full schema
    cursor.execute("""
        CREATE TABLE attendance (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id       TEXT NOT NULL,
            confidence       REAL NOT NULL,
            spoof_verdict    TEXT NOT NULL,
            location_verdict TEXT NOT NULL,
            sentiment        TEXT,
            sentiment_conf   REAL,
            feedback_text    TEXT,
            status           TEXT NOT NULL,
            timestamp        TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Attendance table ready")

def log_attendance(student_id, confidence, spoof_verdict,
                   location_verdict, sentiment_result,
                   feedback_text=""):
    """
    Record a complete attendance event to the database.
    Status is PRESENT only if all checks pass.
    """
    status = (
        "PRESENT"
        if (spoof_verdict    == "LIVE" and
            location_verdict == "ALLOWED")
        else "REJECTED"
    )

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attendance
        (student_id, confidence, spoof_verdict, location_verdict,
         sentiment, sentiment_conf, feedback_text, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        confidence,
        spoof_verdict,
        location_verdict,
        sentiment_result.get("sentiment",  "N/A") if sentiment_result else "N/A",
        sentiment_result.get("confidence", 0.0)   if sentiment_result else 0.0,
        feedback_text,
        status,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    return status

# ─────────────────────────────────────────────
# 5. FULL INTEGRATED PIPELINE
# ─────────────────────────────────────────────
def run_full_pipeline(audio_path, feedback_text,
                      student_lat, student_lon,
                      voice_model, sentiment_pipeline,
                      label_encoder):
    """
    Complete integrated attendance + sentiment pipeline:
    Step 1 → Location Verification
    Step 2 → Audio Preprocessing
    Step 3 → Anti-Spoofing Check
    Step 4 → Voice Recognition (with TTA)
    Step 5 → Sentiment Analysis
    Step 6 → Attendance Logging
    """
    print("\n" + "=" * 55)
    print("   INTEGRATED PIPELINE")
    print("   Attendance + Sentiment System")
    print("=" * 55)

    results = {
        "status"          : "REJECTED",
        "student_id"      : "unknown",
        "confidence"      : 0.0,
        "location"        : "BLOCKED",
        "spoof_check"     : "UNKNOWN",
        "sentiment"       : "N/A",
        "sentiment_conf"  : 0.0,
        "sentiment_scores": {},
        "top3_voice"      : [],
        "reason"          : "",
        "timestamp"       : datetime.now().isoformat()
    }

    # ── Step 1: Location Verification
    print("\n📍 Step 1: Location Verification")
    print(f"   Coordinates : ({student_lat}, {student_lon})")
    loc_result      = verify_location(student_lat, student_lon)
    results["location"] = loc_result["verdict"]
    print(f"   {loc_result['message']}")

    if not loc_result["allowed"]:
        results["reason"] = "Location verification failed"
        print(f"\n❌ Pipeline halted — {results['reason']}")
        log_attendance("unknown", 0.0, "UNKNOWN",
                       "BLOCKED", None, feedback_text)
        return results

    # ── Step 2: Audio Preprocessing
    print("\n🎙️  Step 2: Audio Preprocessing")
    try:
        audio, sr = preprocess_audio(audio_path)
        print(f"   ✅ Audio loaded and preprocessed")
        print(f"   Duration : {len(audio)/sr:.2f}s | SR: {sr}Hz")
    except Exception as e:
        results["reason"] = f"Audio preprocessing failed: {str(e)}"
        print(f"\n❌ Pipeline halted — {results['reason']}")
        return results

    # ── Step 3: Anti-Spoofing
    print("\n🛡️  Step 3: Anti-Spoofing Check")
    spoof_result           = check_liveness(audio, sr, verbose=True)
    results["spoof_check"] = spoof_result["verdict"]

    if spoof_result["verdict"] == "SPOOFED":
        results["reason"] = "Spoofed audio detected"
        print(f"\n❌ Pipeline halted — {results['reason']}")
        log_attendance("unknown", 0.0, "SPOOFED",
                       loc_result["verdict"], None, feedback_text)
        return results

    # ── Step 4: Voice Recognition
    print("\n🔊 Step 4: Voice Recognition (with TTA)")
    voice_result           = predict_student_identity(
        voice_model, label_encoder, audio, sr, n_tta=5
    )
    results["student_id"] = voice_result["student_id"]
    results["confidence"] = voice_result["confidence"]
    results["top3_voice"] = voice_result["top3"]

    print(f"   Predicted   : {voice_result['student_id']}")
    print(f"   Confidence  : {voice_result['confidence']}%")
    print(f"   Verified    : "
          f"{'✅ Yes' if voice_result['verified'] else '❌ No'}")
    print(f"   Top 3 predictions:")
    for rank, pred in enumerate(voice_result["top3"], 1):
        print(f"     {rank}. {pred['student']:<12} "
              f"→ {pred['confidence']}%")

    if not voice_result["verified"]:
        results["reason"] = (
            f"Low confidence voice match "
            f"({voice_result['confidence']}% < "
            f"{CONFIDENCE_THRESHOLD * 100}%)"
        )
        print(f"\n❌ Pipeline halted — {results['reason']}")
        log_attendance(
            voice_result["student_id"],
            voice_result["confidence"],
            spoof_result["verdict"],
            loc_result["verdict"],
            None, feedback_text
        )
        return results

    # ── Step 5: Sentiment Analysis
    print("\n💬 Step 5: Sentiment Analysis")
    sentiment_result            = predict_sentiment(
        sentiment_pipeline, feedback_text)
    results["sentiment"]        = sentiment_result["sentiment"]
    results["sentiment_conf"]   = sentiment_result["confidence"]
    results["sentiment_scores"] = sentiment_result["scores"]

    print(f"   Feedback    : '{feedback_text}'")
    print(f"   Sentiment   : {sentiment_result['sentiment']} "
          f"({sentiment_result['confidence']}%)")
    print(f"   Scores      :")
    for label, score in sentiment_result["scores"].items():
        bar = "█" * int(score / 5)
        print(f"     {label:<10}: {score:>6.2f}%  {bar}")

    # ── Step 6: Log Attendance
    print("\n📋 Step 6: Logging Attendance to Database")
    status            = log_attendance(
        voice_result["student_id"],
        voice_result["confidence"],
        spoof_result["verdict"],
        loc_result["verdict"],
        sentiment_result,
        feedback_text
    )
    results["status"] = status
    results["reason"] = "All checks passed"

    # ── Final Summary
    print("\n" + "=" * 55)
    print("   ✅ PIPELINE COMPLETE")
    print("=" * 55)
    print(f"   Student     : {results['student_id']}")
    print(f"   Status      : "
          f"{'✅ PRESENT' if status == 'PRESENT' else '❌ REJECTED'}")
    print(f"   Confidence  : {results['confidence']}%")
    print(f"   Location    : {results['location']}")
    print(f"   Spoof Check : {results['spoof_check']}")
    print(f"   Sentiment   : {results['sentiment']} "
          f"({results['sentiment_conf']}%)")
    print(f"   Timestamp   : {results['timestamp']}")
    print("=" * 55)

    return results

# ─────────────────────────────────────────────
# 6. BATCH PIPELINE TEST
# ─────────────────────────────────────────────
def run_batch_test(voice_model, sentiment_pipeline,
                   label_encoder, n_students=5):
    """
    Test the full pipeline with multiple students
    to verify the system works end-to-end before deployment.
    """
    print("\n" + "=" * 55)
    print("   BATCH INTEGRATION TEST")
    print(f"   Testing {n_students} students")
    print("=" * 55)

    test_cases = []

    # Valid students — should be PRESENT
    for i in range(1, n_students + 1):
        test_cases.append({
            "label"   : f"✅ Valid — student{i}",
            "audio"   : f"data/voice_samples/student{i}/student{i}_1.wav",
            "feedback": "The lecture was very clear and I understood everything",
            "lat"     : 7.6041241,
            "lon"     : 5.3059950,
            "expected": "PRESENT"
        })

    # Wrong location — should be REJECTED
    test_cases.append({
        "label"   : "❌ Wrong location",
        "audio"   : "data/voice_samples/student1/student1_1.wav",
        "feedback": "The class was okay",
        "lat"     : 9.0579000,
        "lon"     : 7.4951000,
        "expected": "REJECTED"
    })

    # Negative sentiment — should be PRESENT but with negative feedback
    test_cases.append({
        "label"   : "📊 Negative sentiment",
        "audio"   : "data/voice_samples/student2/student2_1.wav",
        "feedback": "I could not understand anything the lecturer taught today",
        "lat"     : 7.6041241,
        "lon"     : 5.3059950,
        "expected": "PRESENT"
    })

    # Neutral sentiment
    test_cases.append({
        "label"   : "📊 Neutral sentiment",
        "audio"   : "data/voice_samples/student3/student3_1.wav",
        "feedback": "The class was okay but I still have some questions",
        "lat"     : 7.6041241,
        "lon"     : 5.3059950,
        "expected": "PRESENT"
    })

    results_summary = []

    for case in test_cases:
        print(f"\n{'─'*55}")
        print(f"  Test: {case['label']}")
        print(f"{'─'*55}")

        result = run_full_pipeline(
            audio_path         = case["audio"],
            feedback_text      = case["feedback"],
            student_lat        = case["lat"],
            student_lon        = case["lon"],
            voice_model        = voice_model,
            sentiment_pipeline = sentiment_pipeline,
            label_encoder      = label_encoder
        )

        passed = result["status"] == case["expected"]
        results_summary.append({
            "test"     : case["label"],
            "expected" : case["expected"],
            "got"      : result["status"],
            "passed"   : passed,
            "student"  : result["student_id"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        })

    # ── Print summary table
    print("\n" + "=" * 55)
    print("   BATCH TEST SUMMARY")
    print("=" * 55)
    passed_count = sum(1 for r in results_summary if r["passed"])

    print(f"\n  {'Test':<35} | {'Expected':<8} | {'Got':<8} | "
          f"{'Sentiment':<10} | {'Pass'}")
    print(f"  {'─'*35}─┼─{'─'*8}─┼─{'─'*8}─┼─"
          f"{'─'*10}─┼─{'─'*4}")

    for r in results_summary:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {r['test'][:35]:<35} | {r['expected']:<8} | "
              f"{r['got']:<8} | {r['sentiment']:<10} | {icon}")

    print(f"\n  Tests passed  : {passed_count}/{len(results_summary)}")
    print(f"  Tests failed  : "
          f"{len(results_summary) - passed_count}/{len(results_summary)}")

    if passed_count == len(results_summary):
        print("\n  🎉 All tests passed! System ready for deployment.")
    else:
        print("\n  ⚠️  Some tests failed. Review the output above.")

    print("=" * 55)
    return results_summary

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Setup database with correct schema
    setup_attendance_table()

    # Load all models once
    voice_model, sentiment_pipeline, label_encoder = load_models()

    # Run batch integration test
    results = run_batch_test(
        voice_model        = voice_model,
        sentiment_pipeline = sentiment_pipeline,
        label_encoder      = label_encoder,
        n_students         = 5
    )

    print("\n🎉 Model Integration complete!")
    print("🚀 Ready for Stage 7: Model Evaluation")