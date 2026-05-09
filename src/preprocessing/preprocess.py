import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import pandas as pd
import re
import pickle
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VOICE_DIR      = "data/voice_samples"
SENTIMENT_CSV  = "data/sentiment_feedback/transcripts/sentiment_dataset.csv"
OUTPUT_MFCC    = "outputs/features/mfcc"
OUTPUT_SPEC    = "outputs/features/spectrograms"
OUTPUT_TEXT    = "outputs/features/text"
OUTPUT_PLOTS   = "outputs/plots"
SAMPLE_RATE    = 16000
N_MFCC         = 40
MAX_PAD_LENGTH = 200
N_STUDENTS     = 32
N_RECORDINGS   = 7

os.makedirs(OUTPUT_MFCC,  exist_ok=True)
os.makedirs(OUTPUT_SPEC,  exist_ok=True)
os.makedirs(OUTPUT_TEXT,  exist_ok=True)
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

# ─────────────────────────────────────────────
# AUDIO PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────
def load_audio(filepath):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    return audio, sr

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

def remove_silence(audio, sr):
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    intervals        = librosa.effects.split(audio_trimmed, top_db=20)
    audio_clean      = np.concatenate(
        [audio_trimmed[start:end] for start, end in intervals]
    )
    return audio_clean

def pre_emphasis(audio, coeff=0.97):
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def extract_mfcc(audio, sr):
    """
    Extract 40 MFCC features — proven optimal
    for our 32-class speaker recognition task.
    Shape: (40, 200)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LENGTH:
        pad  = MAX_PAD_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LENGTH]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc

def generate_spectrogram(audio, sr, save_path):
    mel_spec    = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax     = plt.subplots(figsize=(4, 3))
    librosa.display.specshow(mel_spec_db, sr=sr,
                              x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    plt.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def preprocess_audio(filepath, sample_rate=SAMPLE_RATE):
    """
    Full preprocessing pipeline.
    Handles WAV, WebM, MP3 and other formats.
    """
    try:
        # ── Try librosa first (handles most formats)
        try:
            import librosa
            audio, sr = librosa.load(
                filepath,
                sr=sample_rate,
                mono=True)

        except Exception as load_err:
            print(f"  ⚠️  librosa load failed: "
                  f"{load_err}")
            # ── Fallback: pydub → numpy
            try:
                from pydub import AudioSegment
                import numpy as np
                seg = AudioSegment.from_file(filepath)
                seg = seg.set_frame_rate(sample_rate)
                seg = seg.set_channels(1)
                samples = seg.get_array_of_samples()
                audio   = np.array(
                    samples, dtype=np.float32)
                audio   = audio / (
                    2 ** (seg.sample_width * 8 - 1))
                sr      = sample_rate
            except Exception as pydub_err:
                print(f"  ❌ pydub fallback "
                      f"failed: {pydub_err}")
                return None, None

        if audio is None or len(audio) == 0:
            return None, None

        # ── Noise reduction
        try:
            import noisereduce as nr
            audio = nr.reduce_noise(
                y=audio, sr=sr,
                prop_decrease=0.8)
        except Exception:
            pass  # skip if noisereduce fails

        # ── Trim silence
        try:
            import librosa
            audio, _ = librosa.effects.trim(
                audio, top_db=20)
        except Exception:
            pass

        # ── Pre-emphasis
        import numpy as np
        audio = np.append(
            audio[0],
            audio[1:] - 0.97 * audio[:-1])

        # ── Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio, sr

    except Exception as e:
        print(f"  ❌ preprocess_audio error: {e}")
        return None, None

# ─────────────────────────────────────────────
# PROCESS ALL VOICE SAMPLES
# ─────────────────────────────────────────────
def process_all_samples():
    print("=" * 55)
    print("   PROCESSING VOICE SAMPLES")
    print("=" * 55)

    all_mfcc   = []
    all_labels = []
    errors     = []

    for i in range(1, N_STUDENTS + 1):
        student_id       = f"student{i}"
        student_dir      = os.path.join(VOICE_DIR, student_id)
        student_mfcc_dir = os.path.join(OUTPUT_MFCC, student_id)
        student_spec_dir = os.path.join(OUTPUT_SPEC, student_id)
        os.makedirs(student_mfcc_dir, exist_ok=True)
        os.makedirs(student_spec_dir, exist_ok=True)

        for j in range(1, N_RECORDINGS + 1):
            filename = f"{student_id}_{j}.wav"
            filepath = os.path.join(student_dir, filename)

            if not os.path.exists(filepath):
                errors.append(f"Missing: {filepath}")
                continue

            try:
                audio, sr = preprocess_audio(filepath)
                mfcc      = extract_mfcc(audio, sr)
                all_mfcc.append(mfcc)
                all_labels.append(student_id)

                np.save(os.path.join(
                    student_mfcc_dir, f"{student_id}_{j}.npy"), mfcc)
                generate_spectrogram(audio, sr, os.path.join(
                    student_spec_dir, f"{student_id}_{j}.png"))

            except Exception as e:
                errors.append(f"Error in {filepath}: {str(e)}")

    print(f"✅ Processed {len(all_mfcc)} recordings")
    print(f"✅ MFCC shape per file : {all_mfcc[0].shape}")

    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
    else:
        print("✅ No errors encountered")

    X  = np.array(all_mfcc)
    le = LabelEncoder()
    y  = le.fit_transform(np.array(all_labels))

    np.save(os.path.join(OUTPUT_MFCC, "X_voice.npy"), X)
    np.save(os.path.join(OUTPUT_MFCC, "y_voice.npy"), y)
    with open(os.path.join(OUTPUT_MFCC, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\n✅ Saved X_voice.npy : shape {X.shape}")
    return X, y, le

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
STOP_WORDS = {
    "i","me","my","we","our","you","your","he","his","she",
    "her","it","its","they","them","their","what","which",
    "who","this","that","these","those","am","is","are","was",
    "were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall",
    "can","need","the","a","an","and","but","or","nor","for",
    "so","yet","both","either","not","no","at","by","from",
    "in","into","of","off","on","out","over","to","up","with"
}

def preprocess_text(text):
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def process_sentiment_text():
    print("\n" + "=" * 55)
    print("   PROCESSING SENTIMENT TEXT DATA")
    print("=" * 55)

    df                      = pd.read_csv(SENTIMENT_CSV)
    df["cleaned_response"]  = df["response"].apply(preprocess_text)
    label_map               = {"Positive": 2, "Negative": 0, "Neutral": 1}
    df["sentiment_encoded"] = df["sentiment_label"].map(label_map)

    output_csv = os.path.join(OUTPUT_TEXT, "sentiment_preprocessed.csv")
    df.to_csv(output_csv, index=False)

    print(f"✅ Total responses : {len(df)}")
    print(f"✅ Saved to        : {output_csv}")
    print(f"\n📊 Distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    return df

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    process_all_samples()
    process_sentiment_text()

    print("\n" + "=" * 55)
    print("✅ Preprocessing complete!")
    print("🚀 Ready for augmentation and model training")
    print("=" * 55)