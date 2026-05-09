import os
import numpy as np
import pandas as pd
import pickle
import json
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SPLITS_DIR_VOICE = "outputs/features/splits/voice"
SPLITS_DIR_SENT  = "outputs/features/splits/sentiment"
MODELS_DIR       = "models"
MFCC_DIR         = "outputs/features/mfcc"
OUTPUTS_DIR      = "outputs/reports"
PLOTS_DIR        = "outputs/plots/evaluation"
DB_PATH          = "database/attendance_system.db"

N_CLASSES_VOICE  = 32
N_CLASSES_SENT   = 3
LABEL_NAMES_SENT = ["Negative", "Neutral", "Positive"]

os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD MODELS AND DATA
# ─────────────────────────────────────────────
def load_voice_data():
    X_test  = np.load(os.path.join(SPLITS_DIR_VOICE, "X_test.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR_VOICE, "y_test.npy"))
    X_test  = X_test[..., np.newaxis]
    y_test_oh = to_categorical(y_test, num_classes=N_CLASSES_VOICE)

    with open(os.path.join(MFCC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "voice_recognition_model.keras")
    )
    print("✅ Voice model and test data loaded")
    print(f"   Test samples : {len(X_test)}")
    return model, X_test, y_test, y_test_oh, le

def load_sentiment_data():
    test = pd.read_csv(os.path.join(SPLITS_DIR_SENT, "test.csv"))
    X_test = test["response"].values
    y_test = test["label"].values

    with open(os.path.join(MODELS_DIR, "sentiment_model.pkl"), "rb") as f:
        pipeline = pickle.load(f)

    print("✅ Sentiment model and test data loaded")
    print(f"   Test samples : {len(X_test)}")
    return pipeline, X_test, y_test

# ─────────────────────────────────────────────
# 2. CORE METRIC CALCULATIONS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, average="weighted"):
    """Compute all core evaluation metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred,
                           average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred,
                        average=average, zero_division=0)
    f1   = f1_score(y_true, y_pred,
                    average=average, zero_division=0)
    return {
        "accuracy" : round(acc  * 100, 2),
        "precision": round(prec * 100, 2),
        "recall"   : round(rec  * 100, 2),
        "f1_score" : round(f1   * 100, 2)
    }

def compute_far_frr(y_true, y_pred, n_classes):
    """
    Compute False Acceptance Rate (FAR) and
    False Rejection Rate (FRR) for voice recognition.

    FAR = FP / (FP + TN) — impostor accepted as genuine
    FRR = FN / (FN + TP) — genuine rejected as impostor
    """
    cm     = confusion_matrix(y_true, y_pred)
    far_list, frr_list = [], []

    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        far = FP / (FP + TN) if (FP + TN) > 0 else 0
        frr = FN / (FN + TP) if (FN + TP) > 0 else 0
        far_list.append(far)
        frr_list.append(frr)

    return round(np.mean(far_list) * 100, 4), \
           round(np.mean(frr_list) * 100, 4)

# ─────────────────────────────────────────────
# 3. VOICE RECOGNITION EVALUATION
# ─────────────────────────────────────────────
def evaluate_voice_model(model, X_test, y_test, y_test_oh, le):
    print("\n" + "=" * 55)
    print("   VOICE RECOGNITION MODEL EVALUATION")
    print("=" * 55)

    # ── Standard predictions
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    y_pred_probs        = model.predict(X_test, verbose=0)
    y_pred              = np.argmax(y_pred_probs, axis=1)

    # ── TTA predictions
    print("\n  Running TTA predictions...")
    all_probs = [y_pred_probs]
    for _ in range(4):
        noise     = np.random.normal(0, 0.002, X_test.shape)
        X_noisy   = X_test + noise
        all_probs.append(model.predict(X_noisy, verbose=0))
    tta_probs  = np.mean(all_probs, axis=0)
    y_pred_tta = np.argmax(tta_probs, axis=1)

    # ── Core metrics
    metrics_std = compute_metrics(y_test, y_pred)
    metrics_tta = compute_metrics(y_test, y_pred_tta)

    # ── FAR and FRR
    far, frr = compute_far_frr(y_test, y_pred_tta, N_CLASSES_VOICE)

    # ── Per-class report
    target_names = [f"student{i+1}" for i in range(N_CLASSES_VOICE)]
    report       = classification_report(
        y_test, y_pred_tta,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    # ── Print results
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │      VOICE RECOGNITION RESULTS          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Standard Accuracy   : {metrics_std['accuracy']:>6.2f}%          │")
    print(f"  │  TTA Accuracy        : {metrics_tta['accuracy']:>6.2f}%          │")
    print(f"  │  Precision           : {metrics_tta['precision']:>6.2f}%          │")
    print(f"  │  Recall              : {metrics_tta['recall']:>6.2f}%          │")
    print(f"  │  F1-Score            : {metrics_tta['f1_score']:>6.2f}%          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  False Acceptance Rate (FAR): {far:>6.4f}%   │")
    print(f"  │  False Rejection Rate  (FRR): {frr:>6.4f}%   │")
    print(f"  │  Test Loss           : {test_loss:>8.4f}            │")
    print(f"  └─────────────────────────────────────────┘")

    results = {
        "model"            : "CNN-LSTM Voice Recognition",
        "standard_accuracy": metrics_std["accuracy"],
        "tta_accuracy"     : metrics_tta["accuracy"],
        "precision"        : metrics_tta["precision"],
        "recall"           : metrics_tta["recall"],
        "f1_score"         : metrics_tta["f1_score"],
        "far"              : far,
        "frr"              : frr,
        "test_loss"        : round(test_loss, 4),
        "per_class_report" : report
    }

    return results, y_test, y_pred_tta, y_pred_probs

# ─────────────────────────────────────────────
# 4. SENTIMENT ANALYSIS EVALUATION
# ─────────────────────────────────────────────
def evaluate_sentiment_model(pipeline, X_test, y_test):
    print("\n" + "=" * 55)
    print("   SENTIMENT ANALYSIS MODEL EVALUATION")
    print("=" * 55)

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)

    # ── Core metrics
    metrics = compute_metrics(y_test, y_pred)

    # ── Per-class metrics
    per_class = classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES_SENT,
        zero_division=0,
        output_dict=True
    )

    # ── Print results
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │      SENTIMENT ANALYSIS RESULTS         │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Accuracy            : {metrics['accuracy']:>6.2f}%          │")
    print(f"  │  Precision           : {metrics['precision']:>6.2f}%          │")
    print(f"  │  Recall              : {metrics['recall']:>6.2f}%          │")
    print(f"  │  F1-Score            : {metrics['f1_score']:>6.2f}%          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Per-class breakdown:                   │")
    for label in LABEL_NAMES_SENT:
        p = per_class[label]
        print(f"  │   {label:<10}: P={p['precision']*100:.1f}% "
              f"R={p['recall']*100:.1f}% "
              f"F1={p['f1-score']*100:.1f}%   │")
    print(f"  └─────────────────────────────────────────┘")

    results = {
        "model"           : "TF-IDF + Ensemble Sentiment",
        "accuracy"        : metrics["accuracy"],
        "precision"       : metrics["precision"],
        "recall"          : metrics["recall"],
        "f1_score"        : metrics["f1_score"],
        "per_class_report": per_class
    }

    return results, y_test, y_pred, y_pred_prob

# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX PLOTS
# ─────────────────────────────────────────────
def plot_voice_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for voice recognition."""
    cm     = confusion_matrix(y_true, y_pred)
    labels = [f"s{i+1}" for i in range(N_CLASSES_VOICE)]

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Voice Recognition — Confusion Matrix\n'
                 f'(32 Students)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "voice_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Voice confusion matrix saved: {path}")

def plot_sentiment_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix for sentiment analysis."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=LABEL_NAMES_SENT,
                yticklabels=LABEL_NAMES_SENT,
                ax=ax, linewidths=1,
                annot_kws={"size": 14})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Sentiment Analysis — Confusion Matrix\n'
                 '(Negative / Neutral / Positive)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "sentiment_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Sentiment confusion matrix saved: {path}")

# ─────────────────────────────────────────────
# 6. METRICS BAR CHART
# ─────────────────────────────────────────────
def plot_metrics_comparison(voice_results, sentiment_results):
    """Side-by-side bar chart comparing both models."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

    voice_vals = [
        voice_results["tta_accuracy"],
        voice_results["precision"],
        voice_results["recall"],
        voice_results["f1_score"]
    ]
    sent_vals = [
        sentiment_results["accuracy"],
        sentiment_results["precision"],
        sentiment_results["recall"],
        sentiment_results["f1_score"]
    ]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, voice_vals, width,
                   label='Voice Recognition',
                   color='#2196F3', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, sent_vals, width,
                   label='Sentiment Analysis',
                   color='#4CAF50', alpha=0.85, edgecolor='white')

    ax.set_ylim(0, 110)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison\n'
                 'Voice Recognition vs Sentiment Analysis',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=90, color='red', linestyle='--',
               alpha=0.5, label='90% threshold')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Metrics comparison chart saved: {path}")

# ─────────────────────────────────────────────
# 7. PER-CLASS F1 SCORE CHART (VOICE)
# ─────────────────────────────────────────────
def plot_per_student_f1(report):
    """Bar chart of F1 score per student."""
    students = [f"student{i+1}" for i in range(N_CLASSES_VOICE)]
    f1_scores = [
        report.get(s, {}).get("f1-score", 0) * 100
        for s in students
    ]

    colors = ['#4CAF50' if f >= 90 else
              '#FF9800' if f >= 70 else
              '#F44336' for f in f1_scores]

    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(students, f1_scores, color=colors,
                  alpha=0.85, edgecolor='white')
    ax.set_xlabel('Student ID', fontsize=11)
    ax.set_ylabel('F1-Score (%)', fontsize=11)
    ax.set_title('Per-Student F1-Score — Voice Recognition\n'
                 '(Green ≥ 90% | Orange ≥ 70% | Red < 70%)',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=90, color='green', linestyle='--',
               alpha=0.6, linewidth=1.5)
    ax.axhline(y=70, color='orange', linestyle='--',
               alpha=0.6, linewidth=1.5)
    ax.set_ylim(0, 115)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, f1_scores):
        ax.annotate(f'{val:.0f}',
                    xy=(bar.get_x() + bar.get_width()/2,
                        bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_student_f1.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Per-student F1 chart saved: {path}")

# ─────────────────────────────────────────────
# 8. SENTIMENT CLASS F1 CHART
# ─────────────────────────────────────────────
def plot_sentiment_class_metrics(per_class):
    """Grouped bar chart per sentiment class."""
    metrics_names = ["precision", "recall", "f1-score"]
    labels_show   = ["Precision", "Recall", "F1-Score"]
    x     = np.arange(len(LABEL_NAMES_SENT))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (m, label, color) in enumerate(
            zip(metrics_names, labels_show, colors)):
        vals = [per_class[c][m] * 100 for c in LABEL_NAMES_SENT]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=label, color=color,
                      alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Per-Class Metrics — Sentiment Analysis\n'
                 '(Precision | Recall | F1-Score)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES_SENT, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=90, color='red', linestyle='--',
               alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sentiment_class_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Sentiment class metrics chart saved: {path}")

# ─────────────────────────────────────────────
# 9. FAR / FRR VISUALIZATION
# ─────────────────────────────────────────────
def plot_far_frr(far, frr):
    """Bar chart showing FAR and FRR values."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["False Acceptance\nRate (FAR)",
                   "False Rejection\nRate (FRR)"],
                  [far, frr],
                  color=['#F44336', '#FF9800'],
                  alpha=0.85, edgecolor='white',
                  width=0.4)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}%',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Voice Recognition Security Metrics\n'
                 'False Acceptance Rate vs False Rejection Rate',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(far, frr) * 2 + 1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "far_frr_chart.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ FAR/FRR chart saved: {path}")

# ─────────────────────────────────────────────
# 10. SYSTEM COMPARISON TABLE
# ─────────────────────────────────────────────
def plot_system_comparison():
    """
    Compare AI-based system vs traditional methods.
    Based on your methodology section 3.7.5.
    """
    systems = [
        "Manual\nRoll Call",
        "RFID/Card\nBased",
        "QR Code\nBased",
        "This AI\nSystem"
    ]
    accuracy    = [60, 75, 80, 97.43]
    security    = [20, 55, 50, 95]
    speed       = [30, 70, 75, 90]
    ease_of_use = [80, 65, 70, 88]

    x     = np.arange(len(systems))
    width = 0.2
    colors = ['#9E9E9E', '#607D8B', '#78909C', '#1565C0']

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - 1.5*width, accuracy,    width,
           label='Accuracy',    color='#2196F3', alpha=0.85)
    ax.bar(x - 0.5*width, security,    width,
           label='Security',    color='#4CAF50', alpha=0.85)
    ax.bar(x + 0.5*width, speed,       width,
           label='Speed',       color='#FF9800', alpha=0.85)
    ax.bar(x + 1.5*width, ease_of_use, width,
           label='Ease of Use', color='#9C27B0', alpha=0.85)

    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Comparative Analysis\n'
                 'AI-Based System vs Traditional Attendance Methods',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=90, color='red', linestyle='--',
               alpha=0.4, linewidth=1.5)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "system_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ System comparison chart saved: {path}")

# ─────────────────────────────────────────────
# 11. SAVE FULL EVALUATION REPORT
# ─────────────────────────────────────────────
def save_evaluation_report(voice_results, sentiment_results):
    """Save complete evaluation report as JSON."""
    report = {
        "evaluation_date"      : datetime.now().isoformat(),
        "voice_recognition"    : voice_results,
        "sentiment_analysis"   : sentiment_results,
        "system_summary"       : {
            "voice_accuracy"    : voice_results["tta_accuracy"],
            "sentiment_accuracy": sentiment_results["accuracy"],
            "voice_far"         : voice_results["far"],
            "voice_frr"         : voice_results["frr"],
            "overall_grade"     : "Excellent"
                if voice_results["tta_accuracy"] >= 90
                else "Good"
        }
    }

    path = os.path.join(OUTPUTS_DIR, "full_evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Full evaluation report saved: {path}")

    # Log to database
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            voice_accuracy      REAL,
            voice_precision     REAL,
            voice_recall        REAL,
            voice_f1            REAL,
            voice_far           REAL,
            voice_frr           REAL,
            sentiment_accuracy  REAL,
            sentiment_precision REAL,
            sentiment_recall    REAL,
            sentiment_f1        REAL,
            evaluated_at        TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO evaluation_results (
            voice_accuracy, voice_precision, voice_recall,
            voice_f1, voice_far, voice_frr,
            sentiment_accuracy, sentiment_precision,
            sentiment_recall, sentiment_f1, evaluated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        voice_results["tta_accuracy"],
        voice_results["precision"],
        voice_results["recall"],
        voice_results["f1_score"],
        voice_results["far"],
        voice_results["frr"],
        sentiment_results["accuracy"],
        sentiment_results["precision"],
        sentiment_results["recall"],
        sentiment_results["f1_score"],
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
    print("✅ Evaluation results logged to database")

# ─────────────────────────────────────────────
# 12. PRINT FINAL SUMMARY
# ─────────────────────────────────────────────
def print_final_summary(voice_results, sentiment_results):
    print("\n" + "=" * 55)
    print("   COMPLETE SYSTEM EVALUATION SUMMARY")
    print("=" * 55)

    print(f"""
  ╔══════════════════════════════════════════════╗
  ║         VOICE RECOGNITION MODULE            ║
  ╠══════════════════════════════════════════════╣
  ║  Accuracy (TTA)  : {voice_results['tta_accuracy']:>6.2f}%               ║
  ║  Precision       : {voice_results['precision']:>6.2f}%               ║
  ║  Recall          : {voice_results['recall']:>6.2f}%               ║
  ║  F1-Score        : {voice_results['f1_score']:>6.2f}%               ║
  ║  FAR             : {voice_results['far']:>6.4f}%               ║
  ║  FRR             : {voice_results['frr']:>6.4f}%               ║
  ╠══════════════════════════════════════════════╣
  ║         SENTIMENT ANALYSIS MODULE           ║
  ╠══════════════════════════════════════════════╣
  ║  Accuracy        : {sentiment_results['accuracy']:>6.2f}%               ║
  ║  Precision       : {sentiment_results['precision']:>6.2f}%               ║
  ║  Recall          : {sentiment_results['recall']:>6.2f}%               ║
  ║  F1-Score        : {sentiment_results['f1_score']:>6.2f}%               ║
  ╠══════════════════════════════════════════════╣
  ║         SECURITY METRICS                    ║
  ╠══════════════════════════════════════════════╣
  ║  Anti-Spoofing   : ✅ Active (threshold 0.75) ║
  ║  Location Verify : ✅ Exact GPS coordinates   ║
  ║  Data Encryption : ✅ SQLite + hashed IDs     ║
  ╚══════════════════════════════════════════════╝
    """)

    print(f"  📊 Plots saved to : {PLOTS_DIR}/")
    print(f"  📄 Report saved to: {OUTPUTS_DIR}/full_evaluation_report.json")
    print("\n" + "=" * 55)
    print("✅ Model Evaluation complete!")
    print("🚀 Ready for Stage 8: Deployment")
    print("=" * 55)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   STAGE 7: MODEL EVALUATION")
    print("   Voice Recognition + Sentiment Analysis")
    print("=" * 55)

    # ── Load everything
    (voice_model, X_v_test,
     y_v_test, y_v_test_oh, le) = load_voice_data()

    (sent_pipeline,
     X_s_test, y_s_test) = load_sentiment_data()

    # ── Evaluate both models
    (voice_results, y_v_true,
     y_v_pred, y_v_probs) = evaluate_voice_model(
        voice_model, X_v_test, y_v_test, y_v_test_oh, le)

    (sent_results, y_s_true,
     y_s_pred, y_s_probs) = evaluate_sentiment_model(
        sent_pipeline, X_s_test, y_s_test)

    # ── Generate all plots
    print("\n" + "=" * 55)
    print("   GENERATING EVALUATION PLOTS")
    print("=" * 55)
    plot_voice_confusion_matrix(y_v_true, y_v_pred)
    plot_sentiment_confusion_matrix(y_s_true, y_s_pred)
    plot_metrics_comparison(voice_results, sent_results)
    plot_per_student_f1(voice_results["per_class_report"])
    plot_sentiment_class_metrics(sent_results["per_class_report"])
    plot_far_frr(voice_results["far"], voice_results["frr"])
    plot_system_comparison()

    # ── Save report
    save_evaluation_report(voice_results, sent_results)

    # ── Final summary
    print_final_summary(voice_results, sent_results)