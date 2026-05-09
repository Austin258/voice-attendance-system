import os
import wave
import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VOICE_DIR = "data/voice_samples"
SENTIMENT_CSV = "data/sentiment_feedback/transcripts/sentiment_dataset.csv"
EXPECTED_STUDENTS = 32
EXPECTED_RECORDINGS = 7
EXPECTED_SAMPLE_RATE = 16000

# ─────────────────────────────────────────────
# 1. VALIDATE VOICE SAMPLES
# ─────────────────────────────────────────────
print("=" * 55)
print("   VOICE SAMPLES VALIDATION")
print("=" * 55)

missing_students = []
incomplete_students = []
corrupt_files = []
wrong_sample_rate = []
total_valid = 0

for i in range(1, EXPECTED_STUDENTS + 1):
    student_id = f"student{i}"
    student_dir = os.path.join(VOICE_DIR, student_id)

    # Check folder exists
    if not os.path.exists(student_dir):
        missing_students.append(student_id)
        continue

    # Check recordings
    wav_files = [f for f in os.listdir(student_dir) if f.endswith(".wav")]

    if len(wav_files) < EXPECTED_RECORDINGS:
        incomplete_students.append({
            "student": student_id,
            "found": len(wav_files),
            "expected": EXPECTED_RECORDINGS
        })

    # Validate each wav file
    for wav_file in wav_files:
        filepath = os.path.join(student_dir, wav_file)
        try:
            with wave.open(filepath, 'r') as wf:
                sample_rate = wf.getframerate()
                if sample_rate < EXPECTED_SAMPLE_RATE:
                    wrong_sample_rate.append({
                        "file": filepath,
                        "sample_rate": sample_rate
                    })
                else:
                    total_valid += 1
        except Exception as e:
            corrupt_files.append({"file": filepath, "error": str(e)})

# ── Report ──
print(f"\n✅ Expected Students     : {EXPECTED_STUDENTS}")
print(f"✅ Expected Recordings   : {EXPECTED_RECORDINGS} per student")
print(f"✅ Total Valid Files     : {total_valid}")

if missing_students:
    print(f"\n❌ Missing Student Folders ({len(missing_students)}):")
    for s in missing_students:
        print(f"   - {s}")
else:
    print(f"\n✅ All {EXPECTED_STUDENTS} student folders found")

if incomplete_students:
    print(f"\n⚠️  Incomplete Recordings ({len(incomplete_students)} students):")
    for s in incomplete_students:
        print(f"   - {s['student']}: {s['found']}/{s['expected']} recordings")
else:
    print(f"✅ All students have {EXPECTED_RECORDINGS} recordings each")

if wrong_sample_rate:
    print(f"\n⚠️  Files with low sample rate ({len(wrong_sample_rate)}):")
    for f in wrong_sample_rate:
        print(f"   - {f['file']} → {f['sample_rate']} Hz")
else:
    print(f"✅ All files meet the {EXPECTED_SAMPLE_RATE} Hz sample rate requirement")

if corrupt_files:
    print(f"\n❌ Corrupt Files ({len(corrupt_files)}):")
    for f in corrupt_files:
        print(f"   - {f['file']}: {f['error']}")
else:
    print("✅ No corrupt files detected")

# ─────────────────────────────────────────────
# 2. VALIDATE SENTIMENT DATASET
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("   SENTIMENT DATASET VALIDATION")
print("=" * 55)

if not os.path.exists(SENTIMENT_CSV):
    print(f"\n❌ Sentiment CSV not found at: {SENTIMENT_CSV}")
else:
    df = pd.read_csv(SENTIMENT_CSV)

    required_columns = ["student_id", "question", "response",
                        "compound_score", "sentiment_label"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    null_counts = df.isnull().sum()
    label_dist = df["sentiment_label"].value_counts()
    valid_labels = {"Positive", "Negative", "Neutral"}
    invalid_labels = set(df["sentiment_label"].unique()) - valid_labels

    print(f"\n✅ Total Responses      : {len(df)}")
    print(f"✅ Columns Found        : {list(df.columns)}")

    if missing_cols:
        print(f"\n❌ Missing Columns: {missing_cols}")
    else:
        print("✅ All required columns present")

    if null_counts.any():
        print(f"\n⚠️  Null Values Detected:")
        print(null_counts[null_counts > 0].to_string())
    else:
        print("✅ No null values found")

    if invalid_labels:
        print(f"\n❌ Invalid Sentiment Labels Found: {invalid_labels}")
    else:
        print("✅ All sentiment labels are valid")

    print(f"\n📊 Sentiment Label Distribution:")
    for label, count in label_dist.items():
        bar = "█" * count
        print(f"   {label:<10}: {count:>3}  {bar}")

# ─────────────────────────────────────────────
# 3. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("   FINAL SUMMARY")
print("=" * 55)

voice_ok = (not missing_students and not incomplete_students
            and not corrupt_files and not wrong_sample_rate)
sentiment_ok = os.path.exists(SENTIMENT_CSV) and not missing_cols and not invalid_labels

if voice_ok and sentiment_ok:
    print("\n🎉 All checks passed! Data is ready for preprocessing.")
else:
    print("\n⚠️  Some issues were found. Please fix them before proceeding.")
    if not voice_ok:
        print("   → Fix voice sample issues listed above")
    if not sentiment_ok:
        print("   → Fix sentiment dataset issues listed above")