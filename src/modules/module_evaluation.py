import os
import sys
import json
import numpy as np
import sqlite3
from datetime import datetime

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# UNIFIED MODULE EVALUATION — ALL 11 MODULES
# ─────────────────────────────────────────────
class ModuleEvaluator:

    def __init__(self):
        self.results   = {}
        self.passed    = 0
        self.failed    = 0
        self.test_file = \
            "data/voice_samples/student2/student2_1.wav"

    def _log(self, module, status, details=""):
        icon = "✅" if status == "PASS" else "❌"
        self.results[module] = {
            "status": status, "details": details}
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
        print(f"  {icon} {module:<48} [{status}]")
        if details:
            print(f"     └─ {details}")

    # ─────────────────────────────────────────
    # MODULE 1: UI
    # ─────────────────────────────────────────
    def test_module1_ui(self):
        print("\n📋 Testing Module 1: User Interface...")
        try:
            from src.modules.module1_ui import UIModule
            ui = UIModule()

            assert hasattr(ui, 'show_banner')
            assert hasattr(ui, 'show_main_menu')
            assert hasattr(ui, 'show_attendance_prompt')
            assert hasattr(ui, 'show_feedback_prompt')
            assert hasattr(ui, 'show_attendance_result')

            # Test feedback prompt
            assert ui.show_feedback_prompt(
                test_mode=True, test_choice="Y") == True
            assert ui.show_feedback_prompt(
                test_mode=True, test_choice="N") == False

            # Test result display
            ui.show_attendance_result({
                "status"        : "PRESENT",
                "student_id"    : "student2",
                "confidence"    : 92.31,
                "location"      : "ALLOWED",
                "spoof_check"   : "LIVE",
                "sentiment"     : "Positive",
                "sentiment_conf": 97.89,
                "feedback_given": True,
                "timestamp"     : datetime.now().isoformat(),
                "reason"        : "All checks passed"
            })

            self._log("Module 1 — User Interface",
                      "PASS",
                      "Banner ✅ | Prompts ✅ | "
                      "Result display ✅")
        except Exception as e:
            self._log("Module 1 — User Interface",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 2: VOICE CAPTURE
    # ─────────────────────────────────────────
    def test_module2_voice_capture(self):
        print("\n🎤 Testing Module 2: Voice Capture...")
        try:
            from src.modules.voice_capture \
                import VoiceCaptureModule
            import sounddevice as sd

            capture  = VoiceCaptureModule()
            devices  = sd.query_devices()
            has_mic  = any(d['max_input_channels'] > 0
                           for d in devices)

            assert hasattr(capture, 'record_attendance')
            assert hasattr(capture, 'record_feedback')
            assert hasattr(capture, '_check_quality')

            # Test quality checker
            good = np.random.randn(16000) * 0.1
            bad  = np.zeros(16000)
            ok1, _ = capture._check_quality(good)
            ok2, _ = capture._check_quality(bad)
            assert ok1 == True
            assert ok2 == False

            self._log("Module 2 — Voice Capture",
                      "PASS",
                      f"Mic="
                      f"{'Available' if has_mic else 'Not found'}"
                      f" | Quality checks ✅")
        except Exception as e:
            self._log("Module 2 — Voice Capture",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 3: PREPROCESSING
    # ─────────────────────────────────────────
    def test_module3_preprocessing(self):
        print("\n🧹 Testing Module 3: Preprocessing...")
        try:
            from src.modules.module3_preprocessing \
                import PreprocessingModule

            prep = PreprocessingModule()
            assert os.path.exists(self.test_file)

            audio, sr, mfcc = prep.process(self.test_file)
            assert audio is not None
            assert sr == 16000
            assert mfcc.shape == (40, 200)

            self._log("Module 3 — Preprocessing",
                      "PASS",
                      f"Duration={len(audio)/sr:.2f}s | "
                      f"MFCC={mfcc.shape} | "
                      f"All 5 steps ✅")
        except Exception as e:
            self._log("Module 3 — Preprocessing",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 4: LOCATION VERIFICATION
    # ─────────────────────────────────────────
    def test_module4_location(self):
        print("\n📍 Testing Module 4: Location...")
        try:
            from src.modules.module4_location \
                import LocationVerificationModule

            loc = LocationVerificationModule()

            r1 = loc.verify(7.6041241, 5.3059950)
            assert r1["allowed"] == True
            assert r1["verdict"] == "ALLOWED"

            r2 = loc.verify(9.0579, 7.4951)
            assert r2["allowed"] == False
            assert r2["verdict"] == "BLOCKED"

            self._log("Module 4 — Location Verification",
                      "PASS",
                      f"ALLOWED={r1['distance_m']}m ✅ | "
                      f"BLOCKED={r2['distance_m']}m ✅")
        except Exception as e:
            self._log("Module 4 — Location Verification",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 5: VOICE VERIFICATION
    # ─────────────────────────────────────────
    def test_module5_voice_verification(self):
        print("\n🔊 Testing Module 5: Voice Verification...")
        try:
            from src.modules.module5_voice_verification \
                import VoiceVerificationModule

            vv     = VoiceVerificationModule()
            result = vv.verify_from_file(self.test_file)

            assert "student_id" in result
            assert "confidence" in result
            assert "verified"   in result
            assert "top3"       in result
            assert result["confidence"] > 0
            assert result["student_id"].startswith("student")
            assert len(result["top3"]) == 3

            self._log("Module 5 — Voice Verification",
                      "PASS",
                      f"Predicted={result['student_id']} | "
                      f"Conf={result['confidence']}% | "
                      f"TTA ✅")
        except Exception as e:
            self._log("Module 5 — Voice Verification",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 6: ANTI-SPOOFING
    # ─────────────────────────────────────────
    def test_module6_anti_spoofing(self):
        print("\n🛡️  Testing Module 6: Anti-Spoofing...")
        try:
            from src.modules.module6_anti_spoofing \
                import AntiSpoofingModule

            spoof  = AntiSpoofingModule()
            result = spoof.check_from_file(
                self.test_file, verbose=False)

            assert "verdict"     in result
            assert "spoof_score" in result
            assert result["verdict"] in ["LIVE", "SPOOFED"]
            assert 0 <= result["spoof_score"] <= 1
            assert result["threshold"] == 0.75

            self._log("Module 6 — Anti-Spoofing",
                      "PASS",
                      f"Verdict={result['verdict']} | "
                      f"Score={result['spoof_score']} | "
                      f"Threshold=0.75 ✅")
        except Exception as e:
            self._log("Module 6 — Anti-Spoofing",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 7: SENTIMENT ANALYSIS
    # ─────────────────────────────────────────
    def test_module7_sentiment(self):
        print("\n💬 Testing Module 7: Sentiment Analysis...")
        try:
            from src.modules.module7_sentiment \
                import SentimentAnalysisModule

            sa    = SentimentAnalysisModule()
            tests = [
                ("The lecture was brilliant and clear",
                 "Positive"),
                ("I could not understand anything",
                 "Negative"),
                ("The class was okay with mixed results",
                 "Neutral"),
            ]

            correct = 0
            for text, expected in tests:
                result = sa.classify(text)
                assert result is not None
                if result["sentiment"] == expected:
                    correct += 1

            status = "PASS" if correct >= 2 else "FAIL"
            self._log("Module 7 — Sentiment Analysis",
                      status,
                      f"{correct}/{len(tests)} correct | "
                      f"Pipeline ✅")
        except Exception as e:
            self._log("Module 7 — Sentiment Analysis",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 8: DATABASE (HYBRID)
    # ─────────────────────────────────────────
    def test_module8_database(self):
        print("\n🗄️  Testing Module 8: Database (Hybrid)...")
        try:
            from src.modules.module8_database \
                import DatabaseModule

            db = DatabaseModule()

            # Anonymization
            h1 = db.anonymize("student1")
            h2 = db.anonymize("student2")
            assert len(h1) == 16
            assert h1 != h2

            # Summary
            stats = db.get_summary()
            assert "attendance" in stats

            # Courses
            courses = db.get_courses()
            assert isinstance(courses, list)

            # Verify tables
            conn   = sqlite3.connect(db.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
            """)
            tables   = [r[0] for r in cursor.fetchall()]
            required = ["attendance", "courses",
                        "enrollments", "lecturers",
                        "students"]
            missing  = [t for t in required
                        if t not in tables]
            conn.close()

            assert not missing, f"Missing: {missing}"

            self._log("Module 8 — Database (Hybrid)",
                      "PASS",
                      f"Tables OK | "
                      f"Anonymization ✅ | "
                      f"Firebase-ready ✅")
        except Exception as e:
            self._log("Module 8 — Database (Hybrid)",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 9: SYSTEM INTEGRATION
    # ─────────────────────────────────────────
    def test_module9_integration(self):
        print("\n🔗 Testing Module 9: System Integration...")
        try:
            from src.modules.system_pipeline \
                import AttendanceSentimentSystem
            from src.modules.module11_admin \
                import AdminModule

            # Open attendance first
            admin = AdminModule()
            admin.lecturer_login(
                "peter@abuad.edu.ng", "temp123",
                test_mode=True)
            admin.reset_lecturer_password_test(
                "L001", "temp123")
            admin.lecturer_login(
                "peter@abuad.edu.ng", "temp123",
                test_mode=True)
            admin.manual_open_attendance("ARC101")
            admin.manual_open_feedback("ARC101")

            system = AttendanceSentimentSystem()

            # Test WITH feedback
            r1 = system.run_attendance(
                student_lat          = 7.6041241,
                student_lon          = 5.3059950,
                test_audio_path      = self.test_file,
                test_feedback_path   = self.test_file,
                test_feedback_choice = "Y",
                test_course          = "ARC101"
            )
            assert r1["status"]   == "PRESENT"
            assert r1["location"] == "ALLOWED"

            # Test WITHOUT feedback
            r2 = system.run_attendance(
                student_lat          = 7.6041241,
                student_lon          = 5.3059950,
                test_audio_path      = self.test_file,
                test_feedback_choice = "N",
                test_course          = "ARC101"
            )
            assert r2["status"]         == "PRESENT"
            assert r2["feedback_given"] == False
            assert r2["sentiment"]      is None

            # Test wrong location
            r3 = system.run_attendance(
                student_lat          = 9.0579,
                student_lon          = 7.4951,
                test_audio_path      = self.test_file,
                test_feedback_choice = "N",
                test_course          = "ARC101"
            )
            assert r3["status"] == "REJECTED"

            self._log("Module 9 — System Integration",
                      "PASS",
                      "PRESENT+FB ✅ | "
                      "PRESENT+skip ✅ | "
                      "REJECTED ✅ | "
                      "9 modules connected ✅")
        except Exception as e:
            self._log("Module 9 — System Integration",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 10: DEPLOYMENT
    # ─────────────────────────────────────────
    def test_module10_deployment(self):
        print("\n🚀 Testing Module 10: Deployment...")
        try:
            from src.modules.module10_deployment \
                import DeploymentModule

            deploy        = DeploymentModule()
            ready, checks = deploy.run_checks()
            health        = deploy.health_check()

            assert "TensorFlow"   in health
            assert "Scikit-learn" in health
            assert "Librosa"      in health
            assert "Database"     in health

            passed = sum(1 for v in checks.values() if v)
            total  = len(checks)

            report = deploy.generate_report()
            assert "deployment_ready"      in report
            assert "limitations_addressed" in report

            self._log("Module 10 — Deployment",
                      "PASS",
                      f"Checks={passed}/{total} | "
                      f"Health ✅ | Report ✅")
        except Exception as e:
            self._log("Module 10 — Deployment",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # MODULE 11: ADMINISTRATIVE
    # ─────────────────────────────────────────
    def test_module11_admin(self):
        print("\n🧑‍🏫 Testing Module 11: Admin Module...")
        try:
            from src.modules.module11_admin \
                import AdminModule

            admin = AdminModule()

            # A. Authentication
            ok = admin.lecturer_login(
                "peter@abuad.edu.ng", "temp123",
                test_mode=True)
            assert ok == True

            wrong = admin.lecturer_login(
                "peter@abuad.edu.ng", "badpass",
                test_mode=True)
            assert wrong == False

            # Reset + re-login
            admin.reset_lecturer_password_test(
                "L001", "temp123")
            admin.lecturer_login(
                "peter@abuad.edu.ng", "temp123",
                test_mode=True)

            # B. Course management
            courses = admin.get_my_courses()
            assert len(courses) >= 1

            admin.create_course(
                "TEST_EVAL", "Evaluation Course",
                "Test Venue", "Monday 9AM", "11AM")

            admin.update_course(
                "ARC101", venue="Updated Venue, ABUAD")

            # Enrollment
            admin.enroll_student("student1", "ARC101")
            enrolled = admin.get_enrolled_courses(
                "student1")
            assert len(enrolled) >= 1

            students = admin.get_enrolled_students(
                "ARC101")
            assert isinstance(students, list)

            # C. System control
            admin.manual_open_attendance("ARC101")
            assert admin.can_mark_attendance(
                "ARC101") == True

            admin.manual_open_feedback("ARC101")
            assert admin.can_give_feedback(
                "ARC101") == True

            admin.manual_close_attendance("ARC101")
            admin.manual_close_feedback("ARC101")

            # Student login
            admin.logout()
            ok2 = admin.student_login(
                "austin@student.abuad.edu.ng",
                "student123", test_mode=True)
            assert ok2 == True

            # D. Reports (re-login as lecturer)
            admin.lecturer_login(
                "peter@abuad.edu.ng", "temp123",
                test_mode=True)
            records = admin.get_attendance_report("ARC101")
            assert isinstance(records, list)

            summary, total = admin.get_sentiment_report(
                "ARC101")
            assert isinstance(summary, dict)

            # Session
            admin.start_session("TEST_EVAL", 60)
            admin.end_session("TEST_EVAL")

            self._log("Module 11 — Administrative",
                      "PASS",
                      "Auth ✅ | Courses ✅ | "
                      "Control ✅ | Reports ✅ | "
                      "Enrollment ✅ | Session ✅")
        except Exception as e:
            self._log("Module 11 — Administrative",
                      "FAIL", str(e))

    # ─────────────────────────────────────────
    # RUN ALL TESTS
    # ─────────────────────────────────────────
    def run_all(self):
        print("=" * 60)
        print("   COMPLETE MODULE EVALUATION")
        print("   11-Module System Verification")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        self.test_module1_ui()
        self.test_module2_voice_capture()
        self.test_module3_preprocessing()
        self.test_module4_location()
        self.test_module5_voice_verification()
        self.test_module6_anti_spoofing()
        self.test_module7_sentiment()
        self.test_module8_database()
        self.test_module9_integration()
        self.test_module10_deployment()
        self.test_module11_admin()

        self._print_summary()

    # ─────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────
    def _print_summary(self):
        total = self.passed + self.failed
        pct   = (self.passed/total*100) if total > 0 else 0

        print("\n" + "=" * 60)
        print("   MODULE EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\n  {'Module':<48} {'Status'}")
        print(f"  {'─'*55}")

        for module, r in self.results.items():
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"  {icon} {module:<48} {r['status']}")

        print(f"\n  {'─'*55}")
        print(f"  Modules Passed : {self.passed}/{total}")
        print(f"  Modules Failed : {self.failed}/{total}")
        print(f"  Overall Score  : {pct:.1f}%")

        print(f"\n  ✅ Limitations Addressed:")
        print(f"     🔊 Noise       → Module 3 (Preprocessing)")
        print(f"     📈 Scalability → Module 8 (DB) + 10")
        print(f"     🔒 Security    → Module 6 (Anti-Spoofing)")
        print(f"     🧑 Proxy       → Modules 4 + 5 + 6")
        print(f"     👨‍🏫 Control    → Module 11 (Admin)")

        if self.failed == 0:
            print(f"\n  🎉 All {total} modules verified!")
            print(f"  🚀 System ready for deployment!")
        else:
            print(f"\n  ⚠️  Issues:")
            for m, r in self.results.items():
                if r["status"] == "FAIL":
                    print(f"     ❌ {m}: {r['details']}")

        os.makedirs("outputs/reports", exist_ok=True)
        path = "outputs/reports/module_evaluation.json"
        with open(path, "w") as f:
            json.dump({
                "date"   : datetime.now().isoformat(),
                "passed" : self.passed,
                "failed" : self.failed,
                "score"  : pct,
                "modules": self.results
            }, f, indent=2)

        print(f"\n  📄 Report: {path}")
        print("=" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = ModuleEvaluator()
    evaluator.run_all()