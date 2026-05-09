import os
import numpy as np
import pandas as pd
import pickle
import json
import sqlite3
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      GridSearchCV)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SPLITS_DIR  = "outputs/features/splits/sentiment"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs/reports"
DB_PATH     = "database/attendance_system.db"

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

LABEL_MAP   = {0: "Negative", 1: "Neutral", 2: "Positive"}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]

# ─────────────────────────────────────────────
# 1. LOAD SPLITS
# ─────────────────────────────────────────────
def load_splits():
    train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

    train_full = pd.concat([train, val], ignore_index=True)
    full       = pd.concat([train, val, test], ignore_index=True)

    print("✅ Sentiment splits loaded")
    print(f"   Train+Val : {len(train_full)} | Test : {len(test)}")
    print(f"   Full      : {len(full)}")
    print(f"\n   Label distribution (Train+Val):")
    counts = train_full["label"].value_counts().sort_index()
    for k, v in counts.items():
        print(f"   {LABEL_MAP[k]:<10}: {v}")

    return train_full, test, full

# ─────────────────────────────────────────────
# 2. DUAL TFIDF FEATURE UNION
# ─────────────────────────────────────────────
def build_feature_union():
    """
    Combines two TF-IDF vectorizers:
    - Word-level (1,3) ngrams: captures word phrases
    - Char-level (3,5) ngrams: captures spelling patterns
      and subword features that word ngrams miss.
    Together they give the model a richer view of text.
    """
    word_tfidf = TfidfVectorizer(
        max_features  = 2000,
        ngram_range   = (1, 3),
        sublinear_tf  = True,
        min_df        = 1,
        analyzer      = "word",
        strip_accents = "unicode",
        token_pattern = r"\b[a-zA-Z]{2,}\b"
    )

    char_tfidf = TfidfVectorizer(
        max_features  = 1000,
        ngram_range   = (3, 5),
        sublinear_tf  = True,
        min_df        = 1,
        analyzer      = "char_wb"   # char ngrams within word boundaries
    )

    return FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf)
    ])

# ─────────────────────────────────────────────
# 3. BUILD PIPELINE
# ─────────────────────────────────────────────
def build_ensemble_pipeline():
    """
    Dual TF-IDF + 4-model Soft Voting Ensemble:
    - SVM       : best for high-dim sparse text (weight 3)
    - LR        : strong linear baseline (weight 2)
    - GBM       : gradient boosting catches complex patterns (weight 2)
    - RF        : captures non-linear interactions (weight 1)
    """
    features = build_feature_union()

    svm = SVC(
        kernel       = "rbf",
        C            = 50,
        gamma        = "scale",
        probability  = True,
        random_state = 42
    )

    lr = LogisticRegression(
        C            = 15,
        max_iter     = 3000,
        random_state = 42,
        multi_class  = "multinomial",
        solver       = "lbfgs"
    )

    gbm = GradientBoostingClassifier(
        n_estimators  = 200,
        learning_rate = 0.1,
        max_depth     = 4,
        random_state  = 42
    )

    rf = RandomForestClassifier(
        n_estimators      = 400,
        max_depth         = 15,
        min_samples_split = 2,
        random_state      = 42
    )

    ensemble = VotingClassifier(
        estimators = [
            ("svm", svm),
            ("lr",  lr),
            ("gbm", gbm),
            ("rf",  rf)
        ],
        voting  = "soft",
        weights = [3, 2, 2, 1]
    )

    return Pipeline([
        ("features", features),
        ("ensemble", ensemble)
    ])

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────
def cross_validate_model(pipeline, full_df):
    print("\n" + "=" * 55)
    print("   5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 55)

    X      = full_df["response"].values
    y      = full_df["label"].values
    skf    = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=42)
    scores = cross_val_score(pipeline, X, y,
                              cv=skf, scoring="accuracy",
                              n_jobs=-1)

    print(f"\n  Fold Scores   : {[f'{s*100:.2f}%' for s in scores]}")
    print(f"  Mean Accuracy : {scores.mean()*100:.2f}%")
    print(f"  Std Dev       : ±{scores.std()*100:.2f}%")
    return scores

# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
def train_model(pipeline, train_df):
    print("\n" + "=" * 55)
    print("   TRAINING DUAL TFIDF + ENSEMBLE MODEL")
    print("=" * 55)
    print("   Features : Word TF-IDF + Char TF-IDF")
    print("   Models   : SVM + LR + GBM + Random Forest\n")

    X_train = train_df["response"].values
    y_train = train_df["label"].values

    pipeline.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    print(f"  ✅ Training complete")
    print(f"  Training Accuracy : {train_acc * 100:.2f}%")
    return pipeline

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(pipeline, test_df):
    print("\n" + "=" * 55)
    print("   MODEL EVALUATION — TEST SET")
    print("=" * 55)

    X_test = test_df["response"].values
    y_test = test_df["label"].values
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy : {acc * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=LABEL_NAMES,
                                 zero_division=0))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':>12}", end="")
    for name in LABEL_NAMES:
        print(f"{name:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()

    return acc, y_pred

# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
def save_results(pipeline, test_acc, cv_scores):
    model_path = os.path.join(MODELS_DIR, "sentiment_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Model saved to: {model_path}")

    with open(os.path.join(OUTPUTS_DIR,
              "sentiment_training_history.json"), "w") as f:
        json.dump({
            "model"           : "DualTFIDF + SVM+LR+GBM+RF Ensemble v6",
            "cv_scores"       : [float(s) for s in cv_scores],
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std"          : float(cv_scores.std()),
            "test_accuracy"   : float(test_acc),
            "trained_at"      : datetime.now().isoformat()
        }, f, indent=2)
    print(f"✅ Report saved")

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name    TEXT NOT NULL,
            test_accuracy REAL NOT NULL,
            test_loss     REAL NOT NULL,
            epochs_run    INTEGER NOT NULL,
            trained_at    TEXT NOT NULL
        )
    """)
    cursor.execute("""
        INSERT INTO model_results
        (model_name, test_accuracy, test_loss, epochs_run, trained_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        "DualTFIDF_Ensemble_Sentiment_v6",
        float(test_acc), 0.0, 1,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    print("✅ Results logged to database")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   SENTIMENT ANALYSIS MODEL v6")
    print("   Dual TF-IDF + 4-Model Ensemble")
    print("=" * 55)

    train_df, test_df, full_df = load_splits()
    pipeline                   = build_ensemble_pipeline()
    cv_scores                  = cross_validate_model(pipeline, full_df)
    pipeline                   = train_model(pipeline, train_df)
    test_acc, _                = evaluate_model(pipeline, test_df)
    save_results(pipeline, test_acc, cv_scores)

    print("\n" + "=" * 55)
    print(f"✅ Sentiment model v6 complete!")
    print(f"   CV Mean Accuracy  : {cv_scores.mean()*100:.2f}%")
    print(f"   Test Set Accuracy : {test_acc * 100:.2f}%")
    print("🚀 Ready for Stage 6: Model Evaluation")
    print("=" * 55)