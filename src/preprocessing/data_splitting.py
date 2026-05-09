import os
import numpy as np
import pandas as pd
import pickle
import json
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MFCC_DIR     = "outputs/features/mfcc"
TEXT_DIR     = "outputs/features/text"
OUTPUT_DIR   = "outputs/features/splits"
DB_PATH      = "database/attendance_system.db"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
def load_voice_data():
    """Load MFCC features and labels for voice recognition."""
    X = np.load(os.path.join(MFCC_DIR, "X_voice.npy"))
    y = np.load(os.path.join(MFCC_DIR, "y_voice.npy"))

    with open(os.path.join(MFCC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    print(f"✅ Voice data loaded")
    print(f"   X shape : {X.shape}  → (samples, mfcc_coeffs, frames)")
    print(f"   y shape : {y.shape}  → (samples,)")
    print(f"   Classes : {len(le.classes_)} students")

    return X, y, le

def load_sentiment_data():
    """Load preprocessed sentiment text data."""
    csv_path = os.path.join(TEXT_DIR, "sentiment_preprocessed.csv")
    df = pd.read_csv(csv_path)

    print(f"\n✅ Sentiment data loaded")
    print(f"   Total samples : {len(df)}")
    print(f"   Columns       : {list(df.columns)}")

    return df

# ─────────────────────────────────────────────
# 2. SPLIT VOICE DATA  (70 / 15 / 15)
# ─────────────────────────────────────────────
def split_voice_data(X, y):
    """
    Split voice MFCC data into train, validation and test sets.
    Uses stratification to ensure each student is represented
    proportionally across all splits.
    """
    print("\n" + "=" * 55)
    print("   VOICE DATA SPLITTING  (70 / 15 / 15)")
    print("=" * 55)

    # Step 1: Split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Step 2: Split temp into val (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"\n  Total samples  : {len(X)}")
    print(f"  Training set   : {len(X_train)} samples "
          f"({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set : {len(X_val)} samples  "
          f"({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set       : {len(X_test)} samples  "
          f"({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
# 3. SPLIT SENTIMENT DATA  (70 / 15 / 15)
# ─────────────────────────────────────────────
def split_sentiment_data(df):
    """
    Split sentiment text data into train, validation and test sets.
    Stratified by sentiment label to maintain class balance.
    """
    print("\n" + "=" * 55)
    print("   SENTIMENT DATA SPLITTING  (70 / 15 / 15)")
    print("=" * 55)

    X = df["cleaned_response"].values
    y = df["sentiment_encoded"].values

    # Step 1: train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Step 2: val (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"\n  Total samples  : {len(X)}")
    print(f"  Training set   : {len(X_train)} samples "
          f"({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set : {len(X_val)} samples  "
          f"({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set       : {len(X_test)} samples  "
          f"({len(X_test)/len(X)*100:.1f}%)")

    # Label distribution per split
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"\n  📊 Label distribution per split:")
    for split_name, y_split in [("Train", y_train),
                                  ("Val",   y_val),
                                  ("Test",  y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist = {label_map[u]: c for u, c in zip(unique, counts)}
        print(f"     {split_name:<6}: {dist}")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
# 4. SAVE ALL SPLITS
# ─────────────────────────────────────────────
def save_voice_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save voice splits as .npy files."""
    voice_dir = os.path.join(OUTPUT_DIR, "voice")
    os.makedirs(voice_dir, exist_ok=True)

    np.save(os.path.join(voice_dir, "X_train.npy"), X_train)
    np.save(os.path.join(voice_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(voice_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(voice_dir, "y_train.npy"), y_train)
    np.save(os.path.join(voice_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(voice_dir, "y_test.npy"),  y_test)

    print(f"\n✅ Voice splits saved to: {voice_dir}/")
    print(f"   X_train.npy → {X_train.shape}")
    print(f"   X_val.npy   → {X_val.shape}")
    print(f"   X_test.npy  → {X_test.shape}")

def save_sentiment_splits(X_train, X_val, X_test,
                           y_train, y_val, y_test):
    """Save sentiment splits as CSV files."""
    sent_dir = os.path.join(OUTPUT_DIR, "sentiment")
    os.makedirs(sent_dir, exist_ok=True)

    pd.DataFrame({"response": X_train, "label": y_train}).to_csv(
        os.path.join(sent_dir, "train.csv"), index=False)
    pd.DataFrame({"response": X_val,   "label": y_val}).to_csv(
        os.path.join(sent_dir, "val.csv"),   index=False)
    pd.DataFrame({"response": X_test,  "label": y_test}).to_csv(
        os.path.join(sent_dir, "test.csv"),  index=False)

    print(f"\n✅ Sentiment splits saved to: {sent_dir}/")
    print(f"   train.csv → {len(X_train)} samples")
    print(f"   val.csv   → {len(X_val)} samples")
    print(f"   test.csv  → {len(X_test)} samples")

# ─────────────────────────────────────────────
# 5. LOG SPLIT INFO TO DATABASE
# ─────────────────────────────────────────────
def log_splits_to_db(voice_counts, sentiment_counts):
    """Store data split summary in SQLite database."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_splits (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset         TEXT NOT NULL,
            train_count     INTEGER NOT NULL,
            val_count       INTEGER NOT NULL,
            test_count      INTEGER NOT NULL,
            train_pct       REAL NOT NULL,
            val_pct         REAL NOT NULL,
            test_pct        REAL NOT NULL,
            created_at      TEXT NOT NULL
        )
    """)

    now = datetime.now().isoformat()

    for dataset, counts in [("voice", voice_counts),
                             ("sentiment", sentiment_counts)]:
        total = sum(counts)
        cursor.execute("""
            INSERT INTO data_splits
            (dataset, train_count, val_count, test_count,
             train_pct, val_pct, test_pct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset,
            counts[0], counts[1], counts[2],
            round(counts[0]/total*100, 1),
            round(counts[1]/total*100, 1),
            round(counts[2]/total*100, 1),
            now
        ))

    conn.commit()
    conn.close()
    print("\n✅ Split summary logged to database")

# ─────────────────────────────────────────────
# 6. SAVE SPLIT SUMMARY REPORT
# ─────────────────────────────────────────────
def save_summary(voice_shapes, sentiment_counts):
    """Save a JSON summary of all splits."""
    summary = {
        "split_ratio": {
            "train": "70%",
            "validation": "15%",
            "test": "15%"
        },
        "voice_recognition": {
            "X_train": list(voice_shapes[0]),
            "X_val"  : list(voice_shapes[1]),
            "X_test" : list(voice_shapes[2])
        },
        "sentiment_analysis": {
            "train": sentiment_counts[0],
            "val"  : sentiment_counts[1],
            "test" : sentiment_counts[2]
        },
        "random_seed"  : RANDOM_SEED,
        "stratified"   : True,
        "created_at"   : datetime.now().isoformat()
    }

    out_path = os.path.join(OUTPUT_DIR, "split_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Split summary saved to: {out_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DATA SPLITTING  —  70 / 15 / 15")
    print("=" * 55)

    # Load data
    X_v, y_v, le = load_voice_data()
    df_sent       = load_sentiment_data()

    # Split voice data
    (X_v_train, X_v_val, X_v_test,
     y_v_train, y_v_val, y_v_test) = split_voice_data(X_v, y_v)

    # Split sentiment data
    (X_s_train, X_s_val, X_s_test,
     y_s_train, y_s_val, y_s_test) = split_sentiment_data(df_sent)

    # Save splits
    save_voice_splits(X_v_train, X_v_val, X_v_test,
                      y_v_train, y_v_val, y_v_test)

    save_sentiment_splits(X_s_train, X_s_val, X_s_test,
                          y_s_train, y_s_val, y_s_test)

    # Log to DB
    voice_counts     = [len(X_v_train), len(X_v_val), len(X_v_test)]
    sentiment_counts = [len(X_s_train), len(X_s_val), len(X_s_test)]
    log_splits_to_db(voice_counts, sentiment_counts)

    # Save summary
    voice_shapes = [X_v_train.shape, X_v_val.shape, X_v_test.shape]
    save_summary(voice_shapes, sentiment_counts)

    # Final summary
    print("\n" + "=" * 55)
    print("   SPLITTING COMPLETE")
    print("=" * 55)
    print(f"\n  Voice Recognition Splits:")
    print(f"    Train : {len(X_v_train)} samples (70%)")
    print(f"    Val   : {len(X_v_val)} samples (15%)")
    print(f"    Test  : {len(X_v_test)} samples (15%)")
    print(f"\n  Sentiment Analysis Splits:")
    print(f"    Train : {len(X_s_train)} samples (70%)")
    print(f"    Val   : {len(X_s_val)} samples (15%)")
    print(f"    Test  : {len(X_s_test)} samples (15%)")
    print(f"\n🎉 All splits saved and ready for model training!")
    print(f"🚀 Ready for Stage 5: Model Development")
    print("=" * 55)