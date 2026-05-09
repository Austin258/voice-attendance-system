import os
import sys
import time
import re
import numpy as np
import pickle
from datetime import datetime

import tensorflow as tf

sys.path.append(os.path.abspath("."))
from src.preprocessing.preprocess            import preprocess_audio, extract_mfcc
from src.preprocessing.anti_spoofing         import check_liveness
from src.preprocessing.location_verification import verify_location
from src.modules.voice_capture               import VoiceCaptureModule
from src.modules.speech_to_text              import SpeechToTextModule
from src.modules.module8_database            import DatabaseModule
from src.modules.module11_admin              import AdminModule

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODELS_DIR           = "models"
SAMPLE_RATE          = 16000
CONFIDENCE_THRESHOLD = 0.45
FEEDBACK_DURATION    = 15

FEEDBACK_QUESTIONS = [
    "How was today's lecture?",
    "Did you understand the topic covered today?",
    "What challenges did you face during the class?"
]

# ─────────────────────────────────────────────
# MODULE 10: SYSTEM INTEGRATION MODULE
# ─────────────────────────────────────────────
class SystemPipeline:
    """
    System Integration Module.

    Connects ALL modules into 4 unified pipelines:

    ┌─────────────────────────────────────────┐
    │  PIPELINE 1: REGISTRATION               │
    │  UI → Voice Capture → Preprocessing     │
    │  → Database (store student + voice)     │
    ├─────────────────────────────────────────┤
    │  PIPELINE 2: ATTENDANCE                 │
    │  UI → System Control → Location         │
    │  → Voice Capture → Preprocessing        │
    │  → Anti-Spoofing → Voice Verification   │
    │  → Database (store attendance)          │
    ├─────────────────────────────────────────┤
    │  PIPELINE 3: FEEDBACK (Sentiment)       │
    │  UI → System Control → Voice Capture    │
    │  → Preprocessing → Speech-to-Text       │
    │  → Sentiment Analysis → Database        │
    ├─────────────────────────────────────────┤
    │  PIPELINE 4: LECTURER (ADMIN)           │
    │  UI → Admin Module → Course Control     │
    │  → Database → UI Dashboard              │
    └─────────────────────────────────────────┘

    Security layers:
    - Location check   → prevents proxy attendance
    - Anti-Spoofing    → prevents replay attacks
    - Voice Verification → prevents impersonation
    """

    def __init__(self):
        print("=" * 55)
        print("   SYSTEM INTEGRATION MODULE")
        print("   Initializing all pipelines...")
        print("=" * 55)

        self._load_models()
        self.db            = DatabaseModule()
        self.admin         = AdminModule()
        self.voice_capture = VoiceCaptureModule(
            sample_rate=SAMPLE_RATE)
        self.stt           = SpeechToTextModule()

        print("\n✅ All pipelines ready")
        print("=" * 55)

    # ─────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────
    def _load_models(self):
        """Load voice + sentiment models."""
        print("\n📦 Loading models...")

        self.voice_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR,
                         "voice_recognition_model.keras"))
        print("  ✅ Voice recognition model loaded")

        with open(os.path.join(MODELS_DIR,
                  "sentiment_model.pkl"), "rb") as f:
            self.sentiment_pipeline = pickle.load(f)
        print("  ✅ Sentiment analysis model loaded")

        label_encoder_path = "outputs/features/mfcc/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print("  ✅ Label encoder loaded")
        else:
            self.label_encoder = None
            print("  ⚠️  Label encoder not found — voice recognition disabled")

    # ─────────────────────────────────────────
    # SHARED: VOICE CAPTURE
    # ─────────────────────────────────────────
    def _capture_audio(self, student_id="unknown",
                        purpose="attendance",
                        test_filepath=None):
        """Shared voice capture for all pipelines."""
        print(f"\n🎙️  VOICE CAPTURE [{purpose.upper()}]")
        print("─" * 45)

        if test_filepath and os.path.exists(test_filepath):
            print(f"  ℹ️  Test mode: {test_filepath}")
            import soundfile as sf
            audio, sr = sf.read(
                test_filepath, dtype='float32')
            if audio.ndim > 1:
                audio = audio[:, 0]
            print(f"  ✅ Duration: {len(audio)/sr:.2f}s")
            return audio, sr, test_filepath

        if not self.voice_capture.check_microphone():
            return None, None, None

        if purpose == "attendance":
            return self.voice_capture.record_attendance(
                student_id=student_id, max_attempts=3)
        else:
            return self.voice_capture.record_feedback(
                student_id=student_id,
                duration=FEEDBACK_DURATION,
                max_attempts=2)

    # ─────────────────────────────────────────
    # SHARED: PREPROCESSING
    # ─────────────────────────────────────────
    def _preprocess(self, filepath):
        """Shared preprocessing for all pipelines."""
        print(f"\n⚙️  PREPROCESSING")
        print("─" * 45)
        try:
            audio, sr = preprocess_audio(filepath)
            print(f"  ✅ Duration: {len(audio)/sr:.2f}s "
                  f"| SR: {sr}Hz")
            return audio, sr
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return None, None

    # ─────────────────────────────────────────
    # SHARED: ANTI-SPOOFING
    # ─────────────────────────────────────────
    def _anti_spoofing(self, audio, sr):
        """Shared anti-spoofing check."""
        print(f"\n🛡️  ANTI-SPOOFING CHECK")
        print("─" * 45)
        return check_liveness(audio, sr, verbose=True)

    # ─────────────────────────────────────────
    # SHARED: VOICE VERIFICATION
    # ─────────────────────────────────────────
    def _voice_verify(self, audio, sr, n_tta=5):
        """
        Voice Verification (speaker identification).
        Uses CNN-LSTM + Test-Time Augmentation.
        """
        print(f"\n🔊 VOICE VERIFICATION")
        print("─" * 45)

        if self.label_encoder is None:
            print("  ⚠️  Voice recognition disabled — label encoder not found")
            return {
                "student_id": "unknown",
                "confidence": 0.0,
                "verified"  : False,
                "top3"      : [],
                "status"    : "disabled"
            }

        mfcc      = extract_mfcc(audio, sr)
        mfcc_base = mfcc[np.newaxis, ..., np.newaxis]

        all_probs = [self.voice_model.predict(
            mfcc_base, verbose=0)]
        for _ in range(n_tta - 1):
            noise = np.random.normal(
                0, 0.002, mfcc_base.shape)
            all_probs.append(self.voice_model.predict(
                mfcc_base + noise, verbose=0))

        avg_probs  = np.mean(all_probs, axis=0)[0]
        pred_idx   = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        student_id = self.label_encoder.inverse_transform(
            [pred_idx])[0]

        top3 = [
            {
                "student": self.label_encoder
                           .inverse_transform([i])[0],
                "confidence": round(
                    float(avg_probs[i]) * 100, 2)
            }
            for i in np.argsort(avg_probs)[::-1][:3]
        ]

        verified = confidence >= CONFIDENCE_THRESHOLD
        print(f"  Predicted   : {student_id}")
        print(f"  Confidence  : {confidence*100:.2f}%")
        print(f"  Verified    : "
              f"{'✅ Yes' if verified else '❌ No'}")
        print(f"  Top 3:")
        for r, p in enumerate(top3, 1):
            print(f"    {r}. {p['student']:<12} "
                  f"→ {p['confidence']}%")

        return {
            "student_id": student_id,
            "confidence": round(confidence * 100, 2),
            "verified"  : verified,
            "top3"      : top3
        }

    # ─────────────────────────────────────────
    # SHARED: SENTIMENT ANALYSIS
    # ─────────────────────────────────────────
    def _classify_sentiment(self, text):
        """
        Sentiment Analysis pipeline.
        Text → TF-IDF + Ensemble → Pos/Neg/Neutral
        """
        text = re.sub(r"[^a-z\s]", "",
                       text.lower()).strip()
        if not text:
            return None

        probs     = self.sentiment_pipeline.predict_proba(
            [text])[0]
        pred_idx  = int(np.argmax(probs))
        conf      = float(probs[pred_idx])
        label_map = {0: "Negative",
                     1: "Neutral",
                     2: "Positive"}

        result = {
            "sentiment" : label_map[pred_idx],
            "confidence": round(conf * 100, 2),
            "scores"    : {
                "Negative": round(float(probs[0])*100, 2),
                "Neutral" : round(float(probs[1])*100, 2),
                "Positive": round(float(probs[2])*100, 2)
            }
        }

        print(f"\n📊 SENTIMENT ANALYSIS")
        print("─" * 45)
        print(f"  Text      : '{text[:60]}'")
        print(f"  Sentiment : {result['sentiment']} "
              f"({result['confidence']}%)")
        for label, score in result["scores"].items():
            bar = "█" * int(score / 5)
            print(f"    {label:<10}: "
                  f"{score:>6.2f}%  {bar}")

        return result

    # ─────────────────────────────────────────
    # PIPELINE 1: REGISTRATION
    # ─────────────────────────────────────────
    def run_registration(self, student_name,
                          email, matric_number,
                          password,
                          test_audio_path=None):
        """
        PIPELINE 1: Student Registration.

        Flow:
        1. UI      → Student fills form
        2. Capture → Record voice samples
        3. Preproc → Clean + extract features
        4. DB      → Store student + voice template

        This runs ONCE per student.
        """
        start = time.time()

        print("\n" + "=" * 55)
        print("   PIPELINE 1: REGISTRATION")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

        result = {
            "pipeline"  : "registration",
            "status"    : "FAILED",
            "student_id": None,
            "reason"    : ""
        }

        # ── Step 1: Register student in DB
        import sqlite3
        student_id = f"STU_{matric_number}"
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO students
                (student_id, name, email,
                 matric_number, password,
                 role, registered_at)
                VALUES (?, ?, ?, ?, ?, 'student', ?)
            """, (student_id, student_name, email,
                   matric_number,
                   self.db._hash(password),
                   datetime.now().isoformat()))
            conn.commit()
            print(f"\n  ✅ Student registered: "
                  f"{student_name}")
            print(f"     ID     : {student_id}")
            print(f"     Email  : {email}")
            print(f"     Matric : {matric_number}")
        except sqlite3.IntegrityError:
            print(f"  ℹ️  Student already registered")
        finally:
            conn.close()

        result["student_id"] = student_id

        # ── Step 2: Capture voice
        audio, sr, filepath = self._capture_audio(
            student_id = student_id,
            purpose    = "registration",
            test_filepath = test_audio_path
        )

        if audio is None:
            result["reason"] = "Voice capture failed"
            self._print_pipeline_result(result, start)
            return result

        # ── Step 3: Preprocess
        audio, sr = self._preprocess(filepath)
        if audio is None:
            result["reason"] = "Preprocessing failed"
            self._print_pipeline_result(result, start)
            return result

        # ── Step 4: Store voice sample metadata
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO voice_samples
            (student_id, filename, file_path,
             sample_rate, format, recorded_at)
            VALUES (?, ?, ?, 16000, 'wav', ?)
        """, (student_id,
               f"{student_id}_registration.wav",
               test_audio_path or "live_recording",
               datetime.now().isoformat()))
        conn.commit()
        conn.close()

        result["status"] = "SUCCESS"
        result["reason"] = "Registration complete"
        print(f"\n  ✅ Voice template stored")
        self._print_pipeline_result(result, start)
        return result

    # ─────────────────────────────────────────
    # PIPELINE 2: ATTENDANCE
    # ─────────────────────────────────────────
    def run_attendance(self, student_lat,
                        student_lon,
                        student_id="unknown",
                        course_code=None,
                        test_audio_path=None,
                        test_course=None):
        """
        PIPELINE 2: Attendance Marking.

        Flow:
        1. UI        → Student logs in, selects course
        2. SysCtrl   → Check attendance window open
        3. Location  → Verify GPS coordinates
        4. Capture   → Record voice phrase
        5. Preproc   → Clean + extract MFCC
        6. AntiSpoof → Verify voice is live
        7. VoiceVerify → Identify student
        8. DB        → Store attendance record

        Security:
        - Location    → prevents proxy (remote)
        - Anti-Spoof  → prevents replay attacks
        - VoiceVerify → prevents impersonation
        """
        start     = time.time()
        test_mode = test_audio_path is not None

        print("\n" + "=" * 55)
        print("   PIPELINE 2: ATTENDANCE MARKING")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

        result = {
            "pipeline"       : "attendance",
            "status"         : "REJECTED",
            "student_id"     : "unknown",
            "confidence"     : 0.0,
            "location"       : "BLOCKED",
            "spoof_check"    : "UNKNOWN",
            "course_code"    : course_code,
            "course_title"   : None,
            "reason"         : "",
            "processing_time": 0.0,
            "timestamp"      : datetime.now().isoformat()
        }

        # ── Step 1: Course Selection
        course = self._select_course_for_student(
            student_id  = student_id,
            test_mode   = test_mode,
            test_course = test_course or course_code
        )

        if not course:
            result["reason"] = "No course selected"
            self._print_pipeline_result(result, start)
            return result

        result["course_code"]  = course["course_code"]
        result["course_title"] = course["course_title"]

        # ── Step 2: System Control Check
        print(f"\n⚙️  SYSTEM CONTROL CHECK")
        print("─" * 45)

        # Trigger auto-open check
        self.admin.current_lecturer = None
        can_attend = self.db.get_session_status(
            course["course_code"])["attendance_open"]

        if not can_attend:
            result["reason"] = (
                f"Attendance is CLOSED for "
                f"{course['course_code']}. "
                f"Wait for lecturer to open it.")
            print(f"  🔴 {result['reason']}")
            self._print_pipeline_result(result, start)
            return result

        print(f"  🟢 Attendance is OPEN for "
              f"{course['course_code']}")

        # ── Step 3: Location Verification
        print(f"\n📍 LOCATION VERIFICATION")
        print("─" * 45)
        loc = verify_location(
            student_lat, student_lon)
        print(f"  {loc['message']}")
        result["location"] = loc["verdict"]

        if not loc["allowed"]:
            result["reason"] = "Location check failed"
            self.db.log_attendance(
                student_id       = "unknown",
                confidence       = 0.0,
                spoof_verdict    = "UNKNOWN",
                location_verdict = "BLOCKED",
                feedback_given   = False,
                feedback_text    = None,
                sentiment        = None,
                sentiment_conf   = None,
                status           = "REJECTED",
                course_code      = course["course_code"],
                course_title     = course["course_title"]
            )
            self._print_pipeline_result(result, start)
            return result

        # ── Step 4: Voice Capture
        audio, sr, filepath = self._capture_audio(
            student_id    = student_id,
            purpose       = "attendance",
            test_filepath = test_audio_path
        )
        if audio is None:
            result["reason"] = "Voice capture failed"
            self._print_pipeline_result(result, start)
            return result

        # ── Step 5: Preprocessing
        audio, sr = self._preprocess(filepath)
        if audio is None:
            result["reason"] = "Preprocessing failed"
            self._print_pipeline_result(result, start)
            return result

        # ── Step 6: Anti-Spoofing
        spoof = self._anti_spoofing(audio, sr)
        result["spoof_check"] = spoof["verdict"]

        if spoof["verdict"] == "SPOOFED":
            result["reason"] = "Spoofed audio detected"
            self.db.log_attendance(
                student_id       = "unknown",
                confidence       = 0.0,
                spoof_verdict    = "SPOOFED",
                location_verdict = loc["verdict"],
                feedback_given   = False,
                feedback_text    = None,
                sentiment        = None,
                sentiment_conf   = None,
                status           = "REJECTED",
                course_code      = course["course_code"],
                course_title     = course["course_title"]
            )
            self._print_pipeline_result(result, start)
            return result

        # ── Step 7: Voice Verification
        voice = self._voice_verify(audio, sr)
        result["student_id"] = voice["student_id"]
        result["confidence"] = voice["confidence"]

        if not voice["verified"]:
            result["reason"] = (
                f"Low confidence "
                f"({voice['confidence']}%)")
            self.db.log_attendance(
                student_id       = voice["student_id"],
                confidence       = voice["confidence"],
                spoof_verdict    = spoof["verdict"],
                location_verdict = loc["verdict"],
                feedback_given   = False,
                feedback_text    = None,
                sentiment        = None,
                sentiment_conf   = None,
                status           = "REJECTED",
                course_code      = course["course_code"],
                course_title     = course["course_title"]
            )
            self._print_pipeline_result(result, start)
            return result

        # ── Step 8: Store Attendance
        result["status"] = "PRESENT"
        result["reason"] = "All checks passed"

        self.db.log_attendance(
            student_id       = voice["student_id"],
            confidence       = voice["confidence"],
            spoof_verdict    = spoof["verdict"],
            location_verdict = loc["verdict"],
            feedback_given   = False,
            feedback_text    = None,
            sentiment        = None,
            sentiment_conf   = None,
            status           = "PRESENT",
            course_code      = course["course_code"],
            course_title     = course["course_title"]
        )

        print(f"\n💾 DATABASE: Attendance saved")
        print(f"  Student : {voice['student_id']}")
        print(f"  Status  : PRESENT")
        print(f"  Course  : {course['course_code']}")

        self._print_pipeline_result(result, start)
        return result

    # ─────────────────────────────────────────
    # PIPELINE 3: FEEDBACK (SENTIMENT)
    # ─────────────────────────────────────────
    def run_feedback(self, student_id,
                      course_code,
                      test_mode=False,
                      test_feedback_path=None,
                      test_choice=None):
        """
        PIPELINE 3: Feedback + Sentiment Analysis.

        Flow:
        1. UI        → Prompt: "Give feedback?"
        2. SysCtrl   → Check feedback window open
        3. Capture   → Record verbal feedback (5–30s)
        4. Preproc   → Clean audio
        5. STT       → Speech → Text
        6. Sentiment → Classify text (Pos/Neg/Neutral)
        7. DB        → Store sentiment result

        Note: Feedback is OPTIONAL.
        If skipped: sentiment = NULL, text = "Not Provided"
        """
        start = time.time()

        print("\n" + "=" * 55)
        print("   PIPELINE 3: FEEDBACK + SENTIMENT")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

        result = {
            "pipeline"       : "feedback",
            "student_id"     : student_id,
            "course_code"    : course_code,
            "feedback_given" : False,
            "feedback_text"  : None,
            "sentiment"      : None,
            "sentiment_conf" : None,
            "status"         : "SKIPPED",
            "reason"         : ""
        }

        # ── Step 1: UI Prompt
        print(f"\n💬 FEEDBACK PROMPT")
        print("─" * 45)
        print(f"  ✅ Attendance marked successfully!")
        print(f"\n  Would you like to give feedback?")
        print(f"  [Y] Yes — Record (5–30 seconds)")
        print(f"  [N] Skip — No feedback today")

        if test_mode and test_choice is not None:
            wants_feedback = str(
                test_choice).upper() in ["Y", "YES"]
            print(f"  ℹ️  Test choice: {test_choice}")
        else:
            choice = input(
                "\n  Your choice (Y/N): ").strip().upper()
            wants_feedback = choice in ["Y", "YES"]

        if not wants_feedback:
            print(f"\n  ℹ️  Feedback skipped")
            result["reason"] = "Student skipped feedback"
            self._update_attendance_feedback(
                student_id  = student_id,
                course_code = course_code,
                feedback_given = False,
                feedback_text  = "Not Provided",
                sentiment      = None,
                sentiment_conf = None
            )
            return result

        # ── Step 2: System Control Check
        print(f"\n⚙️  SYSTEM CONTROL — FEEDBACK")
        print("─" * 45)
        fb_open = self.db.get_session_status(
            course_code)["feedback_open"]

        if not fb_open:
            print(f"  🔴 Feedback is CLOSED for "
                  f"{course_code}")
            result["reason"] = "Feedback window closed"
            return result

        print(f"  🟢 Feedback is OPEN")

        # ── Step 3: Capture Feedback Audio
        import random
        question = random.choice(FEEDBACK_QUESTIONS)
        print(f"\n  ❓ {question}")
        print(f"  ⏱️  You have {FEEDBACK_DURATION}s")

        audio, sr, filepath = self._capture_audio(
            student_id    = student_id,
            purpose       = "feedback",
            test_filepath = test_feedback_path
        )

        if audio is None or filepath is None:
            print("  ❌ Could not capture feedback")
            result["reason"] = "Audio capture failed"
            return result

        # ── Step 4: Preprocessing
        audio, sr = self._preprocess(filepath)
        if audio is None:
            result["reason"] = "Preprocessing failed"
            return result

        # ── Step 5: Speech-to-Text
        print(f"\n💬 SPEECH-TO-TEXT CONVERSION")
        print("─" * 45)
        text, ok = self.stt.process_feedback_audio(
            test_feedback_path or filepath)
        self.voice_capture.cleanup_temp(filepath)

        if not ok or not text:
            print("  ⚠️  STT failed — "
                  "no internet or unclear audio")
            result["reason"] = "STT conversion failed"
            return result

        # ── Step 6: Sentiment Analysis
        sentiment = self._classify_sentiment(text)

        if sentiment:
            result["feedback_given"] = True
            result["feedback_text"]  = text
            result["sentiment"]      = (
                sentiment["sentiment"])
            result["sentiment_conf"] = (
                sentiment["confidence"])
            result["status"] = "COMPLETED"
            result["reason"] = "Feedback processed"

            # ── Step 7: Update Database
            self._update_attendance_feedback(
                student_id     = student_id,
                course_code    = course_code,
                feedback_given = True,
                feedback_text  = text,
                sentiment      = sentiment["sentiment"],
                sentiment_conf = sentiment["confidence"]
            )

            print(f"\n💾 DATABASE: Feedback saved")
            print(f"  Student   : {student_id}")
            print(f"  Sentiment : {sentiment['sentiment']}")
            print(f"  Confidence: {sentiment['confidence']}%")

        elapsed = round(time.time() - start, 2)
        result["processing_time"] = elapsed
        return result

    # ─────────────────────────────────────────
    # PIPELINE 4: LECTURER (ADMIN)
    # ─────────────────────────────────────────
    def run_lecturer_pipeline(self,
                               lecturer_email,
                               lecturer_password,
                               course_code=None,
                               action="dashboard",
                               test_mode=False):
        """
        PIPELINE 4: Lecturer Admin Pipeline.

        Flow:
        1. UI     → Lecturer logs in
        2. Admin  → Access owned courses only
        3. Admin  → Control session (open/close)
        4. DB     → Retrieve attendance + sentiment
        5. UI     → Display dashboard

        Actions: dashboard, open_att, close_att,
                 open_fb, close_fb, end_session
        """
        print("\n" + "=" * 55)
        print("   PIPELINE 4: LECTURER ADMIN")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 55)

        # ── Step 1: Login
        ok = self.admin.login(
            lecturer_email, lecturer_password,
            test_mode=test_mode)

        if not ok:
            print("  ❌ Login failed")
            return False

        # ── Step 2: View courses + select
        if not course_code:
            course = self.admin.select_course(
                test_mode=test_mode)
            if not course:
                return False
            course_code = course["course_code"]

        # ── Step 3: Execute action
        print(f"\n  📋 Action: {action.upper()}")

        if action == "dashboard":
            self.admin.show_dashboard(course_code)

        elif action == "open_att":
            self.admin.manual_open_attendance(
                course_code)

        elif action == "close_att":
            self.admin.manual_close_attendance(
                course_code)

        elif action == "open_fb":
            self.admin.manual_open_feedback(
                course_code)

        elif action == "close_fb":
            self.admin.manual_close_feedback(
                course_code)

        elif action == "end_session":
            self.admin.end_session(course_code)

        elif action == "start_session":
            self.admin.start_session(
                course_code, test_mode=test_mode)

        elif action == "reports":
            today = datetime.now().strftime("%Y-%m-%d")
            self.admin.print_attendance_report(
                course_code, today)
            self.admin.print_sentiment_report(
                course_code, today)
            self.admin.print_monitoring_report(
                course_code)

        self.admin.logout()
        return True

    # ─────────────────────────────────────────
    # FULL PIPELINE: ATTENDANCE + FEEDBACK
    # ─────────────────────────────────────────
    def run_full(self, student_lat, student_lon,
                  student_id="unknown",
                  course_code=None,
                  test_audio_path=None,
                  test_feedback_path=None,
                  test_feedback_choice=None,
                  test_course=None):
        """
        Run attendance pipeline, then feedback pipeline.
        This is the complete student experience.
        """
        # Run attendance
        att_result = self.run_attendance(
            student_lat    = student_lat,
            student_lon    = student_lon,
            student_id     = student_id,
            course_code    = course_code,
            test_audio_path= test_audio_path,
            test_course    = test_course
        )

        # Only run feedback if attendance passed
        if att_result["status"] == "PRESENT":
            fb_result = self.run_feedback(
                student_id         = att_result["student_id"],
                course_code        = att_result["course_code"],
                test_mode          = test_audio_path is not None,
                test_feedback_path = test_feedback_path,
                test_choice        = test_feedback_choice
            )
            att_result["feedback"] = fb_result
        else:
            att_result["feedback"] = None

        return att_result

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _select_course_for_student(self, student_id,
                                    test_mode=False,
                                    test_course=None):
        """Show enrolled courses for a student."""
        # Get enrolled or all courses
        if student_id and student_id != "unknown":
            courses = self.db.get_enrolled_courses(
                student_id)
        else:
            courses = self.db.get_courses()

        if not courses:
            # Fallback to all courses
            courses = self.db.get_courses()

        print("\n" + "=" * 55)
        print("  📚 SELECT YOUR COURSE")
        print("=" * 55)
        print(f"  {'#':<4} {'Code':<10} "
              f"{'Title':<28} {'Status'}")
        print(f"  {'─'*55}")

        for idx, c in enumerate(courses, 1):
            status = self.db.get_session_status(c[1])
            att    = "🟢" if status["attendance_open"] \
                     else "🔴"
            print(f"  {idx:<4} {c[1]:<10} "
                  f"{c[2][:27]:<28} {att}")

        print(f"\n  🟢 Att Open  🔴 Att Closed")

        if test_mode and test_course:
            print(f"  ℹ️  Test: selecting {test_course}")
            for c in courses:
                if c[1] == str(test_course).upper():
                    return {
                        "course_code" : c[1],
                        "course_title": c[2],
                        "venue"       : c[3],
                        "start_time"  : c[4],
                        "end_time"    : c[5]
                    }
            # If not found, return first available
            if courses:
                c = courses[0]
                return {
                    "course_code" : c[1],
                    "course_title": c[2],
                    "venue"       : c[3],
                    "start_time"  : c[4],
                    "end_time"    : c[5]
                }
            return None

        try:
            choice = int(input(
                f"\n  Select (1-{len(courses)}): "))
            if 1 <= choice <= len(courses):
                c = courses[choice - 1]
                return {
                    "course_code" : c[1],
                    "course_title": c[2],
                    "venue"       : c[3],
                    "start_time"  : c[4],
                    "end_time"    : c[5]
                }
        except (ValueError, IndexError):
            pass

        return None

    def _update_attendance_feedback(self, student_id,
                                     course_code,
                                     feedback_given,
                                     feedback_text,
                                     sentiment,
                                     sentiment_conf):
        """Update the latest attendance record with feedback."""
        import sqlite3
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE attendance
            SET feedback_given = ?,
                feedback_text  = ?,
                sentiment      = ?,
                sentiment_conf = ?
            WHERE student_id = ?
              AND course_code = ?
              AND id = (
                  SELECT id FROM attendance
                  WHERE student_id = ?
                    AND course_code = ?
                  ORDER BY timestamp DESC
                  LIMIT 1
              )
        """, (
            1 if feedback_given else 0,
            feedback_text,
            sentiment,
            sentiment_conf,
            student_id, course_code.upper(),
            student_id, course_code.upper()
        ))
        conn.commit()
        conn.close()

    def _print_pipeline_result(self, result, start):
        """Print formatted pipeline result."""
        elapsed = round(time.time() - start, 2)
        result["processing_time"] = elapsed

        pipeline = result.get("pipeline", "").upper()
        status   = result.get("status", "UNKNOWN")
        icon     = "✅" if status in [
            "PRESENT", "SUCCESS"] else "❌"

        print("\n" + "=" * 55)
        print(f"   {pipeline} PIPELINE RESULT")
        print("=" * 55)
        print(f"  Status    : {icon} {status}")

        if "student_id" in result:
            print(f"  Student   : {result['student_id']}")
        if "confidence" in result:
            print(f"  Confidence: {result['confidence']}%")
        if "location" in result:
            print(f"  Location  : {result['location']}")
        if "spoof_check" in result:
            print(f"  Spoof     : {result['spoof_check']}")
        if "course_code" in result and result["course_code"]:
            print(f"  Course    : {result['course_code']}")
        if result.get("reason"):
            print(f"  Reason    : {result['reason']}")

        print(f"  Time      : {elapsed}s")
        print(f"  Timestamp : {result.get('timestamp', datetime.now().isoformat())}")
        print("=" * 55)


# ─────────────────────────────────────────────
# ENTRY POINT — TEST ALL PIPELINES
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   SYSTEM PIPELINE TEST")
    print("   All 4 Pipelines")
    print("=" * 55)

    pipeline = SystemPipeline()

    # ── Open attendance for testing
    print("\n🔓 Setting up test session...")
    pipeline.admin.login(
        "peter@demo.com", "temp123", test_mode=True)
    pipeline.admin.start_session(
        "CSC308", test_mode=True)
    pipeline.admin.manual_open_attendance("CSC308")
    pipeline.admin.manual_open_feedback("CSC308")
    pipeline.admin.current_lecturer = None

    # ── Pipeline 1: Registration
    print("\n" + "─"*55)
    print("🧪 PIPELINE 1: Registration Test")
    print("─"*55)
    pipeline.run_registration(
        student_name   = "Test Student",
        email          = "test@student.com",
        matric_number  = "TEST001",
        password       = "test123",
        test_audio_path= "data/voice_samples/"
                         "student1/student1_1.wav"
    )

    # ── Pipeline 2: Attendance WITH feedback
    print("\n" + "─"*55)
    print("🧪 PIPELINE 2+3: Attendance + Feedback")
    print("─"*55)
    result = pipeline.run_full(
        student_lat          = 7.6041241,
        student_lon          = 5.3059950,
        student_id           = "STU001",
        test_audio_path      = "data/voice_samples/"
                               "student2/student2_1.wav",
        test_feedback_path   = "data/voice_samples/"
                               "student2/student2_1.wav",
        test_feedback_choice = "Y",
        test_course          = "CSC308"
    )

    # ── Pipeline 2: Attendance WITHOUT feedback
    print("\n" + "─"*55)
    print("🧪 PIPELINE 2+3: Attendance (no feedback)")
    print("─"*55)
    pipeline.run_full(
        student_lat          = 7.6041241,
        student_lon          = 5.3059950,
        student_id           = "STU002",
        test_audio_path      = "data/voice_samples/"
                               "student3/student3_1.wav",
        test_feedback_choice = "N",
        test_course          = "CSC308"
    )

    # ── Pipeline 2: Wrong location (REJECT)
    print("\n" + "─"*55)
    print("🧪 PIPELINE 2: Wrong Location (REJECT)")
    print("─"*55)
    pipeline.run_full(
        student_lat          = 9.0579000,
        student_lon          = 7.4951000,
        student_id           = "STU003",
        test_audio_path      = "data/voice_samples/"
                               "student1/student1_1.wav",
        test_feedback_choice = "N",
        test_course          = "CSC308"
    )

    # ── Pipeline 4: Lecturer Admin
    print("\n" + "─"*55)
    print("🧪 PIPELINE 4: Lecturer Dashboard")
    print("─"*55)
    pipeline.run_lecturer_pipeline(
        lecturer_email    = "peter@demo.com",
        lecturer_password = "temp123",
        course_code       = "CSC308",
        action            = "dashboard",
        test_mode         = True
    )

    print("\n" + "=" * 55)
    print("✅ All Pipelines Complete!")
    print("=" * 55)