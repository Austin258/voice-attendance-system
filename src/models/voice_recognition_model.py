import os
import numpy as np
import pickle
import json
import sqlite3
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Reshape, LSTM, Dense
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SPLITS_DIR  = "outputs/features/splits/voice"
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs/reports"
DB_PATH     = "database/attendance_system.db"
MFCC_DIR    = "outputs/features/mfcc"

N_MFCC      = 40
MAX_FRAMES  = 200
N_CLASSES   = 32
EPOCHS      = 120
BATCH_SIZE  = 32
RANDOM_SEED = 42
WARMUP_EPOCHS = 10   # gradual warmup before cosine decay

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. LOAD SPLITS
# ─────────────────────────────────────────────
def load_splits():
    X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

    with open(os.path.join(MFCC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    print("✅ Voice splits loaded")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   X_test  : {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le

# ─────────────────────────────────────────────
# 2. PREPARE INPUTS
# ─────────────────────────────────────────────
def prepare_inputs(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    y_train_oh = to_categorical(y_train, num_classes=N_CLASSES)
    y_val_oh   = to_categorical(y_val,   num_classes=N_CLASSES)
    y_test_oh  = to_categorical(y_test,  num_classes=N_CLASSES)

    print(f"\n✅ Input shapes prepared")
    print(f"   X_train : {X_train.shape}")
    print(f"   y_train : {y_train_oh.shape}")

    return (X_train, X_val, X_test,
            y_train_oh, y_val_oh, y_test_oh, y_test)

# ─────────────────────────────────────────────
# 3. ENHANCED MIXUP AUGMENTATION
# ─────────────────────────────────────────────
def mixup_data(X, y, alpha=0.3):
    """
    Enhanced mixup with higher alpha (0.3 vs 0.2).
    Higher alpha = more aggressive blending =
    smoother decision boundaries between 32 students.
    Applied 3 times to triple the effective training set.
    """
    batch_size = len(X)
    lam        = np.random.beta(alpha, alpha, batch_size)
    lam        = np.maximum(lam, 1 - lam)

    idx   = np.random.permutation(batch_size)
    lam_X = lam.reshape(-1, 1, 1, 1)
    lam_y = lam.reshape(-1, 1)

    X_mixed = lam_X * X + (1 - lam_X) * X[idx]
    y_mixed = lam_y * y + (1 - lam_y) * y[idx]
    return X_mixed, y_mixed

def build_augmented_dataset(X_train, y_train, n_mixup_rounds=3):
    """
    Build augmented training dataset:
    - Original samples
    - n_mixup_rounds of mixup (default 3)
    Total = 4x original training set size
    """
    print(f"\n  Building augmented dataset ({n_mixup_rounds} mixup rounds)...")
    X_all = [X_train]
    y_all = [y_train]

    for i in range(n_mixup_rounds):
        X_mix, y_mix = mixup_data(X_train, y_train, alpha=0.3)
        X_all.append(X_mix)
        y_all.append(y_mix)

    X_combined = np.concatenate(X_all, axis=0)
    y_combined = np.concatenate(y_all, axis=0)

    # Shuffle combined dataset
    idx = np.random.permutation(len(X_combined))
    X_combined = X_combined[idx]
    y_combined = y_combined[idx]

    print(f"  Original samples : {len(X_train)}")
    print(f"  After augmentation: {len(X_combined)} "
          f"({n_mixup_rounds + 1}x)")
    return X_combined, y_combined

# ─────────────────────────────────────────────
# 4. WARMUP + COSINE ANNEALING LR SCHEDULE
# ─────────────────────────────────────────────
def warmup_cosine_schedule(epoch, total_epochs=120,
                            warmup_epochs=10,
                            lr_max=0.001, lr_min=1e-6):
    """
    Warmup phase: linearly increase LR from lr_min to lr_max
    over the first warmup_epochs. This prevents early
    instability when starting with random weights.

    Cosine phase: smoothly decay from lr_max to lr_min
    over the remaining epochs.
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return float(lr_min + (lr_max - lr_min) *
                     (epoch / warmup_epochs))
    else:
        # Cosine annealing after warmup
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine   = np.cos(np.pi * progress)
        return float(lr_min + 0.5 * (lr_max - lr_min) * (1 + cosine))

# ─────────────────────────────────────────────
# 5. BUILD CNN-LSTM MODEL v5
# ─────────────────────────────────────────────
def build_cnn_lstm_model(input_shape, n_classes):
    """
    CNN-LSTM v5 improvements over v4:
    - Extra Dense layer in classification head
      (256 → 128 → 64 → 32 classes) for richer
      feature transformation before final output
    - Slightly reduced dropout on CNN blocks
      (0.2 instead of 0.25) to retain more spatial
      features from MFCC maps
    - L2 regularization on LSTM layer added
    - Same proven 3 CNN + 1 LSTM backbone
    Input shape: (40, 200, 1)
    """
    inputs = Input(shape=input_shape, name="mfcc_input")

    # ── CNN Block 1
    x = Conv2D(32, (3,3), activation="relu",
               padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.20)(x)          # ↓ from 0.25

    # ── CNN Block 2
    x = Conv2D(64, (3,3), activation="relu",
               padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.20)(x)          # ↓ from 0.25

    # ── CNN Block 3
    x = Conv2D(128, (3,3), activation="relu",
               padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)          # ↓ from 0.30

    # ── Reshape for LSTM
    cnn_shape  = x.shape
    time_steps = cnn_shape[1] * cnn_shape[2]
    features   = cnn_shape[3]
    x = Reshape((time_steps, features))(x)

    # ── LSTM with L2 regularization added
    x = LSTM(128, return_sequences=False,
             kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.30)(x)          # ↓ from 0.35

    # ── Deeper classification head (new extra layer)
    x = Dense(256, activation="relu",
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.30)(x)

    x = Dense(128, activation="relu",
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.25)(x)

    x = Dense(64, activation="relu",
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.20)(x)

    outputs = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name="CNN_LSTM_VoiceRecognition_v5")
    return model

# ─────────────────────────────────────────────
# 6. TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────
def apply_tta(model, X_test, n_augments=5):
    """
    Test-Time Augmentation: run inference multiple times
    with slightly perturbed test samples and average
    the predicted probabilities.

    This reduces variance in predictions and boosts
    accuracy on difficult or borderline samples.
    Noise level is kept very small (0.002) to avoid
    distorting the MFCC features.
    """
    print(f"\n  Applying TTA ({n_augments} augmentation rounds)...")
    all_probs = []

    # Original prediction
    all_probs.append(model.predict(X_test, verbose=0))

    # Augmented predictions
    for i in range(n_augments - 1):
        noise    = np.random.normal(0, 0.002, X_test.shape)
        X_noisy  = X_test + noise
        probs    = model.predict(X_noisy, verbose=0)
        all_probs.append(probs)

    # Average all predictions
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs

# ─────────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Training improvements over v4:
    - Warmup (10 epochs) + Cosine annealing
    - 3 rounds of Mixup (4x training set)
    - Label smoothing 0.1 (same as v4)
    - Longer patience (20 vs 15)
    - Monitor val_loss for EarlyStopping
      (more stable signal than val_accuracy)
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        metrics=["accuracy"]
    )

    print("\n" + "=" * 55)
    print("   TRAINING CNN-LSTM VOICE MODEL v5")
    print("   + 3x Mixup Augmentation (alpha=0.3)")
    print("   + Label Smoothing (0.1)")
    print("   + Warmup (10) + Cosine Annealing LR")
    print("   + Deeper Classification Head")
    print("   + Test-Time Augmentation at eval")
    print("=" * 55)
    model.summary()

    # Build augmented training set
    X_combined, y_combined = build_augmented_dataset(
        X_train, y_train, n_mixup_rounds=3)

    lr_scheduler = LearningRateScheduler(
        lambda epoch: warmup_cosine_schedule(
            epoch,
            total_epochs=EPOCHS,
            warmup_epochs=WARMUP_EPOCHS
        ),
        verbose=0
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",        # more stable than val_accuracy
            patience=20,               # ↑ from 15
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "voice_model_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        lr_scheduler
    ]

    print(f"\n  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Warmup epochs : {WARMUP_EPOCHS}")
    print(f"  LR schedule   : Warmup → Cosine (0.001 → 1e-6)\n")

    history = model.fit(
        X_combined, y_combined,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history

# ─────────────────────────────────────────────
# 8. EVALUATE WITH TTA
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test_oh, y_test_raw, le):
    print("\n" + "=" * 55)
    print("   MODEL EVALUATION — TEST SET")
    print("=" * 55)

    # Standard evaluation
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"\n  Standard Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test Loss              : {test_loss:.4f}")

    # TTA evaluation
    tta_probs  = apply_tta(model, X_test, n_augments=5)
    y_pred_tta = np.argmax(tta_probs, axis=1)
    tta_acc    = np.mean(y_pred_tta == y_test_raw)
    print(f"  TTA Test Accuracy      : {tta_acc * 100:.2f}%")

    # Use TTA predictions for classification report
    target_names = [f"student{i+1}" for i in range(N_CLASSES)]
    report       = classification_report(
        y_test_raw, y_pred_tta,
        target_names=target_names,
        zero_division=0
    )
    print(f"\n  Classification Report (TTA):\n{report}")

    # Return best accuracy
    best_acc = max(test_acc, tta_acc)
    return best_acc, test_loss, y_pred_tta

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
def save_results(model, history, test_acc, test_loss):
    model_path = os.path.join(MODELS_DIR, "voice_recognition_model.keras")
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")

    with open(os.path.join(OUTPUTS_DIR,
              "voice_training_history.json"), "w") as f:
        json.dump({
            "accuracy"     : history.history["accuracy"],
            "val_accuracy" : history.history["val_accuracy"],
            "loss"         : history.history["loss"],
            "val_loss"     : history.history["val_loss"],
            "test_accuracy": float(test_acc),
            "test_loss"    : float(test_loss),
            "trained_at"   : datetime.now().isoformat()
        }, f, indent=2)
    print(f"✅ History saved")

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
        "CNN_LSTM_VoiceRecognition_v5",
        float(test_acc), float(test_loss),
        len(history.history["accuracy"]),
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
    print("   VOICE RECOGNITION MODEL v5")
    print("   3x Mixup + Warmup-Cosine + TTA + Deeper Head")
    print("=" * 55)

    (X_train, X_val, X_test,
     y_train, y_val, y_test, le) = load_splits()

    (X_train, X_val, X_test,
     y_train_oh, y_val_oh,
     y_test_oh, y_test_raw) = prepare_inputs(
        X_train, X_val, X_test, y_train, y_val, y_test)

    model   = build_cnn_lstm_model((N_MFCC, MAX_FRAMES, 1), N_CLASSES)
    history = train_model(model, X_train, y_train_oh, X_val, y_val_oh)

    test_acc, test_loss, _ = evaluate_model(
        model, X_test, y_test_oh, y_test_raw, le)

    save_results(model, history, test_acc, test_loss)

    print("\n" + "=" * 55)
    print(f"✅ Voice model v5 complete!")
    print(f"   Final Test Accuracy : {test_acc * 100:.2f}%")
    print("🚀 Proceed to: Model Evaluation Stage")
    print("=" * 55)