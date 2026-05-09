import os
import numpy as np
import pandas as pd
import pickle
import librosa
import re
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VOICE_DIR    = "data/voice_samples"
MFCC_DIR     = "outputs/features/mfcc"
TEXT_DIR     = "outputs/features/text"
SAMPLE_RATE  = 16000
N_MFCC       = 40
MAX_FRAMES   = 200
N_STUDENTS   = 32
N_RECORDINGS = 7

# ─────────────────────────────────────────────
# AUGMENTATION TECHNIQUES
# ─────────────────────────────────────────────
def add_noise(audio, factor=0.005):
    return audio + factor * np.random.randn(len(audio))

def add_loud_noise(audio, factor=0.015):
    return audio + factor * np.random.randn(len(audio))

def time_stretch_slow(audio):
    return librosa.effects.time_stretch(audio, rate=0.85)

def time_stretch_fast(audio):
    return librosa.effects.time_stretch(audio, rate=1.15)

def pitch_up(audio, sr):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

def pitch_down(audio, sr):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)

def shift_left(audio):
    shift = int(len(audio) * 0.1)
    return np.roll(audio, -shift)

def shift_right(audio):
    shift = int(len(audio) * 0.1)
    return np.roll(audio, shift)

def volume_up(audio):
    return np.clip(audio * 1.5, -1.0, 1.0)

def volume_down(audio):
    return audio * 0.6

def noise_and_pitch(audio, sr):
    a = add_noise(audio)
    return librosa.effects.pitch_shift(a, sr=sr, n_steps=1)

def stretch_and_noise(audio, sr):
    a = librosa.effects.time_stretch(audio, rate=0.9)
    return add_noise(a, factor=0.004)

def pitch_up_stretch(audio, sr):
    a = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
    return librosa.effects.time_stretch(a, rate=0.95)

def noise_shift(audio, sr):
    a = add_noise(audio, factor=0.003)
    return shift_left(a)

# ─────────────────────────────────────────────
# MFCC EXTRACTION (40 coefficients)
# ─────────────────────────────────────────────
def extract_mfcc(audio, sr):
    """
    Extract 40 MFCC features — proven optimal for
    our 32-class speaker recognition task.
    Shape: (40, 200)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_FRAMES:
        pad  = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc

# ─────────────────────────────────────────────
# AUGMENT VOICE DATA
# 15 augmentations × 7 recordings × 32 students
# = 3360 total samples
# ─────────────────────────────────────────────
def augment_voice_data():
    print("=" * 55)
    print("   VOICE DATA AUGMENTATION")
    print("=" * 55)

    augmentations = [
        ("original",          lambda a, sr: a),
        ("noise_low",         lambda a, sr: add_noise(a)),
        ("noise_high",        lambda a, sr: add_loud_noise(a)),
        ("stretch_slow",      lambda a, sr: time_stretch_slow(a)),
        ("stretch_fast",      lambda a, sr: time_stretch_fast(a)),
        ("pitch_up",          lambda a, sr: pitch_up(a, sr)),
        ("pitch_down",        lambda a, sr: pitch_down(a, sr)),
        ("shift_left",        lambda a, sr: shift_left(a)),
        ("shift_right",       lambda a, sr: shift_right(a)),
        ("volume_up",         lambda a, sr: volume_up(a)),
        ("volume_down",       lambda a, sr: volume_down(a)),
        ("noise_and_pitch",   lambda a, sr: noise_and_pitch(a, sr)),
        ("stretch_and_noise", lambda a, sr: stretch_and_noise(a, sr)),
        ("pitch_up_stretch",  lambda a, sr: pitch_up_stretch(a, sr)),
        ("noise_shift",       lambda a, sr: noise_shift(a, sr)),
    ]

    all_mfcc   = []
    all_labels = []
    errors     = []

    for i in range(1, N_STUDENTS + 1):
        student_id  = f"student{i}"
        student_dir = os.path.join(VOICE_DIR, student_id)

        for j in range(1, N_RECORDINGS + 1):
            filepath = os.path.join(
                student_dir, f"{student_id}_{j}.wav")
            if not os.path.exists(filepath):
                errors.append(filepath)
                continue
            try:
                audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                for aug_name, aug_fn in augmentations:
                    try:
                        aug_audio = aug_fn(audio, sr)
                        mfcc      = extract_mfcc(aug_audio, sr)
                        all_mfcc.append(mfcc)
                        all_labels.append(student_id)
                    except Exception:
                        mfcc = extract_mfcc(audio, sr)
                        all_mfcc.append(mfcc)
                        all_labels.append(student_id)
            except Exception as e:
                errors.append(f"{filepath}: {e}")

        if i % 8 == 0 or i == N_STUDENTS:
            print(f"  ✅ {i}/{N_STUDENTS} students | "
                  f"Samples: {len(all_mfcc)}")

    X  = np.array(all_mfcc)
    le = LabelEncoder()
    y  = le.fit_transform(np.array(all_labels))

    os.makedirs(MFCC_DIR, exist_ok=True)
    np.save(os.path.join(MFCC_DIR, "X_voice.npy"), X)
    np.save(os.path.join(MFCC_DIR, "y_voice.npy"), y)
    with open(os.path.join(MFCC_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\n  📊 Voice Augmentation Summary:")
    print(f"     Total samples  : {len(X)}")
    print(f"     Shape          : {X.shape}")
    print(f"     Per student    : {len(X) // N_STUDENTS}")
    if errors:
        print(f"     ⚠️  Errors     : {len(errors)}")
    print(f"\n  ✅ Saved to: {MFCC_DIR}/")
    return X, y, le

# ─────────────────────────────────────────────
# AUGMENT SENTIMENT DATA
# ─────────────────────────────────────────────
def augment_sentiment_data():
    print("\n" + "=" * 55)
    print("   SENTIMENT DATA AUGMENTATION")
    print("=" * 55)

    extra = [
        # ── POSITIVE
        ("The lecture was absolutely brilliant today", "Positive"),
        ("I had no trouble understanding the content", "Positive"),
        ("Superb teaching made this topic very simple", "Positive"),
        ("I grasped everything taught in class today", "Positive"),
        ("Very insightful lecture that clarified a lot", "Positive"),
        ("The class was perfectly paced and very clear", "Positive"),
        ("I walked out of class feeling very confident", "Positive"),
        ("Excellent delivery and very relatable examples", "Positive"),
        ("The topic was simplified in an amazing way", "Positive"),
        ("I thoroughly enjoyed every minute of class", "Positive"),
        ("The lesson flowed smoothly from start to end", "Positive"),
        ("Outstanding session with very helpful visuals", "Positive"),
        ("I could follow every explanation without effort", "Positive"),
        ("The class boosted my confidence in this subject", "Positive"),
        ("Very well structured and easy to understand", "Positive"),
        ("I left class today feeling fully satisfied", "Positive"),
        ("The teaching was top notch and very engaging", "Positive"),
        ("All my doubts were cleared during the lecture", "Positive"),
        ("I appreciated how the topic was broken down", "Positive"),
        ("Fantastic class that exceeded all expectations", "Positive"),
        ("The lecture motivated me to study more today", "Positive"),
        ("Everything was explained with great precision", "Positive"),
        ("I understood even the most complex parts today", "Positive"),
        ("The class was lively and very informative", "Positive"),
        ("I feel fully prepared after today's lecture", "Positive"),
        ("Wonderful experience in class this morning", "Positive"),
        ("The examples resonated with me very well", "Positive"),
        ("I feel empowered to tackle this topic now", "Positive"),
        ("The lecture was inspiring and well delivered", "Positive"),
        ("Best lecture I have attended this semester", "Positive"),
        ("I absorbed everything taught today with ease", "Positive"),
        ("The content was engaging and easy to process", "Positive"),
        ("The lecture answered all my previous questions", "Positive"),
        ("I now have a thorough grasp of this topic", "Positive"),
        ("The class was perfect in every single way", "Positive"),

        # ── NEGATIVE
        ("I completely zoned out during the lecture", "Negative"),
        ("The topic was rushed through without explanation", "Negative"),
        ("I left class more confused than when I arrived", "Negative"),
        ("The lecture was poorly planned and confusing", "Negative"),
        ("I could not grasp a single concept today", "Negative"),
        ("The delivery was monotonous and very boring", "Negative"),
        ("I struggled throughout the entire lecture today", "Negative"),
        ("The class was chaotic and hard to follow", "Negative"),
        ("Nothing was explained clearly during the session", "Negative"),
        ("I felt overwhelmed by the pace of the class", "Negative"),
        ("The session was a complete waste of my time", "Negative"),
        ("I am extremely frustrated after today's class", "Negative"),
        ("The concepts introduced were never fully explained", "Negative"),
        ("I have never been this lost in a lecture before", "Negative"),
        ("The class left me with more questions unanswered", "Negative"),
        ("I struggled to even write down proper notes", "Negative"),
        ("The lecture made no sense to me whatsoever", "Negative"),
        ("I feel demotivated after today's terrible lecture", "Negative"),
        ("The teaching style was very ineffective today", "Negative"),
        ("I could not follow the logic of the explanations", "Negative"),
        ("The class was painfully difficult to sit through", "Negative"),
        ("I need a complete re-teaching of today's topic", "Negative"),
        ("The session was too theoretical with no examples", "Negative"),
        ("Very poor classroom management ruined the lecture", "Negative"),
        ("I was completely disengaged for the whole class", "Negative"),
        ("The lecture skipped too many foundational concepts", "Negative"),
        ("I am deeply confused after today's session", "Negative"),
        ("The content was irrelevant and poorly delivered", "Negative"),
        ("I have no idea what was taught in class today", "Negative"),
        ("The class was the most unproductive session yet", "Negative"),
        ("I felt ignored and lost throughout the lecture", "Negative"),
        ("The explanation style did not work for me at all", "Negative"),
        ("I am disappointed with how today's class went", "Negative"),
        ("The session moved too fast for anyone to follow", "Negative"),
        ("I struggled to stay awake during the dull lecture", "Negative"),

        # ── NEUTRAL
        ("The class was reasonable but nothing outstanding", "Neutral"),
        ("I followed most of it but lost track near the end", "Neutral"),
        ("Today's lecture was okay with a few unclear spots", "Neutral"),
        ("The session was fine but I still have questions", "Neutral"),
        ("I understood the basics but missed the advanced part", "Neutral"),
        ("The class was average with some good moments", "Neutral"),
        ("I got the main idea but need to review my notes", "Neutral"),
        ("The lecture was decent though a bit hard at times", "Neutral"),
        ("Some parts were engaging while others were dull", "Neutral"),
        ("I partially followed the class but need more help", "Neutral"),
        ("The session was acceptable but not very memorable", "Neutral"),
        ("I understood seventy percent of today's content", "Neutral"),
        ("The class was standard without anything surprising", "Neutral"),
        ("I neither loved nor hated today's lecture at all", "Neutral"),
        ("The session was satisfactory but lacked some depth", "Neutral"),
        ("I learned something today but still feel uncertain", "Neutral"),
        ("The class was moderate and somewhat informative", "Neutral"),
        ("I followed along but the ending was a bit unclear", "Neutral"),
        ("The lecture was passable but left me wanting more", "Neutral"),
        ("Some concepts clicked but others still confuse me", "Neutral"),
        ("The class was okay though pacing could improve", "Neutral"),
        ("I got a general understanding but need more review", "Neutral"),
        ("The session was mixed with good and unclear parts", "Neutral"),
        ("I understood the topic at a surface level today", "Neutral"),
        ("The lecture was alright but not particularly helpful", "Neutral"),
        ("I followed most steps but struggled with examples", "Neutral"),
        ("The class covered the topic but lacked elaboration", "Neutral"),
        ("I grasped the concept briefly but need more time", "Neutral"),
        ("The session was fair with room for improvement", "Neutral"),
        ("I understood today's class at a beginner level", "Neutral"),
        ("The lecture was okay though I expected much more", "Neutral"),
        ("I followed the introduction but got lost midway", "Neutral"),
        ("The class was mediocre but had a few good points", "Neutral"),
        ("I understand the basics but advanced parts elude me", "Neutral"),
        ("The lecture was so-so and left a mixed impression", "Neutral"),
    ]

    csv_path = os.path.join(TEXT_DIR, "sentiment_preprocessed.csv")
    df_orig  = pd.read_csv(csv_path)

    label_map  = {"Positive": 2, "Negative": 0, "Neutral": 1}
    extra_rows = []

    for idx, (response, label) in enumerate(extra):
        cleaned = re.sub(r"[^a-z\s]", "",
                         response.lower()).strip()
        extra_rows.append({
            "student_id"       : f"student{(idx % 32) + 1}",
            "question"         : "How was today's lecture?",
            "response"         : response,
            "compound_score"   : 0.0,
            "sentiment_label"  : label,
            "cleaned_response" : cleaned,
            "sentiment_encoded": label_map[label]
        })

    df_extra    = pd.DataFrame(extra_rows)
    df_combined = pd.concat([df_orig, df_extra], ignore_index=True)
    df_combined.drop_duplicates(subset=["response"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    out_path = os.path.join(TEXT_DIR, "sentiment_preprocessed.csv")
    df_combined.to_csv(out_path, index=False)

    print(f"  📊 Sentiment Augmentation Summary:")
    print(f"     Original  : {len(df_orig)}")
    print(f"     Added     : {len(df_extra)}")
    print(f"     Total     : {len(df_combined)}")
    print(f"\n  📊 Label Distribution:")
    print(df_combined["sentiment_label"].value_counts().to_string())
    print(f"\n  ✅ Saved to: {out_path}")
    return df_combined

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    augment_voice_data()
    augment_sentiment_data()

    print("\n" + "=" * 55)
    print("✅ Augmentation complete!")
    print("🚀 Now re-run splitting and retrain both models")
    print("=" * 55)