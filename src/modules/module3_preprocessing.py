import os
import sys
import numpy as np
import librosa
import noisereduce as nr

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# MODULE 3: PREPROCESSING MODULE
# ─────────────────────────────────────────────
class PreprocessingModule:
    """
    Prepares raw audio data for machine learning models.

    Addresses Noise Sensitivity limitation by:
    - Spectral noise reduction
    - Silence removal
    - Pre-emphasis filtering
    - Normalization
    - MFCC feature extraction

    All steps follow the methodology in Section 3.4.
    """

    def __init__(self, sample_rate=16000,
                 n_mfcc=40, max_frames=200):
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc
        self.max_frames  = max_frames

    # ─────────────────────────────────────────
    # STEP 1: LOAD AUDIO
    # ─────────────────────────────────────────
    def load_audio(self, filepath):
        audio, sr = librosa.load(filepath,
                                  sr=self.sample_rate)
        return audio, sr

    # ─────────────────────────────────────────
    # STEP 2: NOISE REDUCTION
    # Addresses: Noise Sensitivity
    # ─────────────────────────────────────────
    def reduce_noise(self, audio, sr):
        """
        Spectral subtraction noise reduction.
        Removes background noise common in classrooms.
        """
        return nr.reduce_noise(
            y=audio, sr=sr, prop_decrease=0.8)

    # ─────────────────────────────────────────
    # STEP 3: SILENCE REMOVAL
    # ─────────────────────────────────────────
    def remove_silence(self, audio, sr):
        """
        Trim leading/trailing silence and
        eliminate unnecessary pauses.
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        intervals  = librosa.effects.split(trimmed, top_db=20)
        if len(intervals) == 0:
            return trimmed
        return np.concatenate(
            [trimmed[s:e] for s, e in intervals])

    # ─────────────────────────────────────────
    # STEP 4: PRE-EMPHASIS FILTER
    # ─────────────────────────────────────────
    def pre_emphasis(self, audio, coeff=0.97):
        """Amplify high-frequency components."""
        return np.append(
            audio[0], audio[1:] - coeff * audio[:-1])

    # ─────────────────────────────────────────
    # STEP 5: NORMALIZATION
    # ─────────────────────────────────────────
    def normalize(self, audio):
        """Normalize amplitude to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    # ─────────────────────────────────────────
    # STEP 6: MFCC EXTRACTION
    # Addresses: Noise Sensitivity
    # ─────────────────────────────────────────
    def extract_mfcc(self, audio, sr):
        """
        Extract 40 MFCC coefficients.
        MFCCs model human auditory perception and
        capture unique speaker voice characteristics.
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc)

        if mfcc.shape[1] < self.max_frames:
            pad  = self.max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad)),
                          mode='constant')
        else:
            mfcc = mfcc[:, :self.max_frames]

        mfcc = ((mfcc - np.mean(mfcc)) /
                (np.std(mfcc) + 1e-9))
        return mfcc

    # ─────────────────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────────────────
    def process(self, filepath):
        """
        Full preprocessing pipeline:
        Load → Denoise → Silence Remove →
        Pre-emphasis → Normalize → MFCC
        """
        audio, sr = self.load_audio(filepath)
        audio     = self.reduce_noise(audio, sr)
        audio     = self.remove_silence(audio, sr)
        audio     = self.pre_emphasis(audio)
        audio     = self.normalize(audio)
        mfcc      = self.extract_mfcc(audio, sr)
        return audio, sr, mfcc

    def process_audio_array(self, audio, sr):
        """Process numpy audio array directly."""
        audio = self.reduce_noise(audio, sr)
        audio = self.remove_silence(audio, sr)
        audio = self.pre_emphasis(audio)
        audio = self.normalize(audio)
        mfcc  = self.extract_mfcc(audio, sr)
        return audio, sr, mfcc


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 3: PREPROCESSING TEST")
    print("=" * 55)

    preproc   = PreprocessingModule()
    test_file = "data/voice_samples/student1/student1_1.wav"

    if os.path.exists(test_file):
        audio, sr, mfcc = preproc.process(test_file)
        print(f"✅ Audio processed")
        print(f"   Duration  : {len(audio)/sr:.2f}s")
        print(f"   MFCC shape: {mfcc.shape}")
    else:
        print("⚠️  Test file not found")