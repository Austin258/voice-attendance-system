import os
import sys
import sqlite3
import hashlib
import json
import threading
from datetime import datetime, timedelta

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# FIREBASE CONFIGURATION
# ─────────────────────────────────────────────
FIREBASE_ENABLED     = True
FIREBASE_CREDENTIALS = "serviceAccountKey.json"
FIREBASE_DB_URL      = "https://voice-attendance-system-35d9c-default-rtdb.europe-west1.firebasedatabase.app/"

# ─────────────────────────────────────────────
# MODULE 8: DATABASE MODULE (SQLite + Firebase)
# ─────────────────────────────────────────────
class DatabaseModule:
    """
    Single source of truth for ALL database tables.

    Tables:
    - lecturers     : lecturer accounts
    - students      : student accounts
    - courses       : courses owned by lecturers
    - enrollments   : student ↔ course relationships
    - attendance    : attendance records
    - feedback      : sentiment feedback records
    - voice_samples : voice recording metadata
    - location_log  : GPS verification logs
    - system_control: session open/close control
    - model_results : ML model performance
    - data_splits   : ML training splits
    - evaluation_results: evaluation metrics
    - access_log    : system access log

    Sample Logins:
    ─────────────────────────────────────────
    LECTURERS:
      peter@demo.com  / temp123  (Dr. Peter)
      anna@demo.com   / temp123  (Dr. Anna)
      james@demo.com  / temp123  (Dr. James)
      grace@demo.com  / temp123  (Dr. Grace)
      david@demo.com  / temp123  (Dr. David)

    STUDENTS:
      austin@student.com / 123456 (Austin)
      john@student.com   / 123456 (John)
      mary@student.com   / 123456 (Mary)
      paul@student.com   / 123456 (Paul)
      linda@student.com  / 123456 (Linda)
    ─────────────────────────────────────────

    Hybrid Storage:
    - SQLite  : local, offline, always available
    - Firebase: cloud sync when internet available
    """

    DB_PATH = "database/attendance_system.db"

    def __init__(self):
        os.makedirs("database", exist_ok=True)
        self._setup_all_tables()
        self._seed_sample_data()
        self.firebase_db = None
        self._init_firebase()

    # ─────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────
    def _hash(self, text):
        """SHA-256 hash for passwords and IDs."""
        return hashlib.sha256(text.encode()).hexdigest()

    def anonymize(self, student_id):
        """Anonymize student ID for privacy."""
        return self._hash(str(student_id))[:16]

    def _get_conn(self):
        """
        Get a database connection with:
        - WAL journal mode (prevents locking)
        - 30 second timeout (waits instead of failing)
        - Thread-safe check_same_thread=False
        """
        conn = sqlite3.connect(
            self.DB_PATH,
            timeout         = 30,
            check_same_thread = False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # ─────────────────────────────────────────
    # SETUP ALL TABLES
    # ─────────────────────────────────────────
    def _setup_all_tables(self):
        """Create ALL system tables."""
        conn   = self._get_conn()
        cursor = conn.cursor()

        # ── Lecturers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lecturers (
                lecturer_id  TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                email        TEXT NOT NULL UNIQUE,
                password     TEXT NOT NULL,
                role         TEXT DEFAULT 'lecturer',
                first_login  INTEGER DEFAULT 1,
                created_at   TEXT NOT NULL
            )
        """)

        # ── Students
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id    TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                matric_number TEXT,
                password      TEXT NOT NULL,
                role          TEXT DEFAULT 'student',
                registered_at TEXT NOT NULL
            )
        """)

        # ── Courses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code  TEXT NOT NULL UNIQUE,
                course_title TEXT NOT NULL,
                venue        TEXT NOT NULL,
                start_time   TEXT NOT NULL,
                end_time     TEXT NOT NULL,
                lecturer_id  TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                day          TEXT NOT NULL DEFAULT 'Monday',
                FOREIGN KEY (lecturer_id)
                    REFERENCES lecturers(lecturer_id)
            )
        """)

        # ── Enrollments (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id    TEXT NOT NULL,
                course_id     INTEGER NOT NULL,
                joined_at     TEXT NOT NULL,
                UNIQUE(student_id, course_id),
                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),
                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id)
            )
        """)

        # ── Attendance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id       TEXT NOT NULL,
                course_id        INTEGER,
                course_code      TEXT,
                course_title     TEXT,
                confidence       REAL NOT NULL,
                spoof_verdict    TEXT NOT NULL,
                location_verdict TEXT NOT NULL,
                feedback_given   INTEGER DEFAULT 0,
                feedback_text    TEXT,
                sentiment        TEXT,
                sentiment_conf   REAL,
                status           TEXT NOT NULL,
                synced_firebase  INTEGER DEFAULT 0,
                timestamp        TEXT NOT NULL,
                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),
                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id)
            )
        """)

        # ── Feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                course_id   INTEGER,
                course_code TEXT,
                sentiment   TEXT,
                text        TEXT,
                confidence  REAL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (student_id)
                    REFERENCES students(student_id),
                FOREIGN KEY (course_id)
                    REFERENCES courses(course_id)
            )
        """)

        # ── Voice samples
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT NOT NULL,
                filename    TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                sample_rate INTEGER DEFAULT 16000,
                format      TEXT DEFAULT 'wav',
                recorded_at TEXT NOT NULL
            )
        """)

        # ── Location log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                lat        REAL,
                lon        REAL,
                distance_m REAL,
                verdict    TEXT,
                allowed    INTEGER,
                timestamp  TEXT
            )
        """)

        # ── System control
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_control (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id           INTEGER,
                course_code         TEXT NOT NULL,
                lecturer_id         TEXT NOT NULL,
                attendance_open     INTEGER DEFAULT 0,
                feedback_open       INTEGER DEFAULT 0,
                auto_open_time      TEXT,
                auto_close_time     TEXT,
                manually_overridden INTEGER DEFAULT 0,
                session_started_at  TEXT,
                session_ended_at    TEXT,
                date                TEXT NOT NULL
            )
        """)

        # ── Sentiment feedback (standalone ML)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_feedback (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id        TEXT NOT NULL,
                course_code       TEXT,
                question          TEXT,
                response          TEXT,
                compound_score    REAL,
                sentiment_label   TEXT,
                sentiment_encoded INTEGER,
                collected_at      TEXT
            )
        """)

        # ── Model results
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

        # ── Data splits
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_splits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset     TEXT NOT NULL,
                train_count INTEGER NOT NULL,
                val_count   INTEGER NOT NULL,
                test_count  INTEGER NOT NULL,
                train_pct   REAL NOT NULL,
                val_pct     REAL NOT NULL,
                test_pct    REAL NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)

        # ── Evaluation results
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

        # ── Access log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                action     TEXT NOT NULL,
                table_name TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                status     TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # SEED SAMPLE DATA
    # ─────────────────────────────────────────
    def _seed_sample_data(self):
        """
        Insert sample lecturers, students, courses
        and enrollments for testing.

        LECTURER LOGINS:
          peter@demo.com  / temp123
          anna@demo.com   / temp123
          james@demo.com  / temp123
          grace@demo.com  / temp123
          david@demo.com  / temp123

        STUDENT LOGINS:
          austin@student.com / 123456
          john@student.com   / 123456
          mary@student.com   / 123456
          paul@student.com   / 123456
          linda@student.com  / 123456
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        now    = datetime.now().isoformat()

        # ── Seed Lecturers
        lecturers = [
            ("L001", "Dr. Peter", "peter@demo.com",
             self._hash("temp123")),
            ("L002", "Dr. Anna",  "anna@demo.com",
             self._hash("temp123")),
            ("L003", "Dr. James", "james@demo.com",
             self._hash("temp123")),
            ("L004", "Dr. Grace", "grace@demo.com",
             self._hash("temp123")),
            ("L005", "Dr. David", "david@demo.com",
             self._hash("temp123")),
        ]
        for l in lecturers:
            cursor.execute("""
                INSERT OR IGNORE INTO lecturers
                (lecturer_id, name, email, password,
                 role, first_login, created_at)
                VALUES (?, ?, ?, ?, 'lecturer', 1, ?)
            """, (*l, now))

        # ── Seed Students
        students = [
            ("STU001", "Austin",
             "austin@student.com",
             "MAT/2021/001",
             self._hash("123456")),
            ("STU002", "John",
             "john@student.com",
             "MAT/2021/002",
             self._hash("123456")),
            ("STU003", "Mary",
             "mary@student.com",
             "MAT/2021/003",
             self._hash("123456")),
            ("STU004", "Paul",
             "paul@student.com",
             "MAT/2021/004",
             self._hash("123456")),
            ("STU005", "Linda",
             "linda@student.com",
             "MAT/2021/005",
             self._hash("123456")),
        ]
        for s in students:
            cursor.execute("""
                INSERT OR IGNORE INTO students
                (student_id, name, email,
                 matric_number, password,
                 role, registered_at)
                VALUES (?, ?, ?, ?, ?,
                        'student', ?)
            """, (*s, now))

        # ── Seed Courses (linked to lecturers)
        courses = [
            ("CSC308", "Software Engineering",
             "Computer Science Dept, ABUAD",
             "08:00", "10:00", "Monday", "L001"),
            ("CSC306", "Database Management Systems",
             "Computer Science Dept, ABUAD",
             "10:00", "12:00", "Tuesday", "L002"),
            ("CSC318", "Artificial Intelligence",
             "Computer Science Dept, ABUAD",
             "12:00", "14:00", "Wednesday", "L003"),
            ("CSC320", "Computer Networks",
             "Computer Science Dept, ABUAD",
             "14:00", "16:00", "Thursday", "L004"),
            ("CSC322", "Operating Systems",
             "Computer Science Dept, ABUAD",
             "08:00", "10:00", "Friday", "L005"),
        ]
        for c in courses:
            cursor.execute("""
                INSERT OR IGNORE INTO courses
                (course_code, course_title, venue,
                 start_time, end_time, day,
                 lecturer_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (*c, now, now))

        conn.commit()

        # ── Seed Enrollments
        # Get course IDs first
        cursor.execute(
            "SELECT course_id, course_code FROM courses")
        course_map = {
            row[1]: row[0]
            for row in cursor.fetchall()
        }

        # Each student enrolled in 2–3 courses
        enrollments = [
            # Austin → CSC308, CSC306, CSC318
            ("STU001", "CSC308"),
            ("STU001", "CSC306"),
            ("STU001", "CSC318"),
            # John → CSC308, CSC320, CSC322
            ("STU002", "CSC308"),
            ("STU002", "CSC320"),
            ("STU002", "CSC322"),
            # Mary → CSC306, CSC318, CSC320
            ("STU003", "CSC306"),
            ("STU003", "CSC318"),
            ("STU003", "CSC320"),
            # Paul → CSC308, CSC306, CSC322
            ("STU004", "CSC308"),
            ("STU004", "CSC306"),
            ("STU004", "CSC322"),
            # Linda → CSC318, CSC320, CSC322
            ("STU005", "CSC318"),
            ("STU005", "CSC320"),
            ("STU005", "CSC322"),
        ]
        for student_id, course_code in enrollments:
            course_id = course_map.get(course_code)
            if course_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO enrollments
                    (student_id, course_id, joined_at)
                    VALUES (?, ?, ?)
                """, (student_id, course_id, now))

        conn.commit()
        conn.close()

    def reset_system_data(self):
        """Reset the database back to the seeded sample state."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        now    = datetime.now().isoformat()

        sample_lecturers = [
            ("L001", "Dr. Peter", "peter@demo.com",
             self._hash("temp123")),
            ("L002", "Dr. Anna",  "anna@demo.com",
             self._hash("temp123")),
            ("L003", "Dr. James", "james@demo.com",
             self._hash("temp123")),
            ("L004", "Dr. Grace", "grace@demo.com",
             self._hash("temp123")),
            ("L005", "Dr. David", "david@demo.com",
             self._hash("temp123")),
        ]

        sample_students = [
            ("STU001", "Austin",
             "austin@student.com",
             "MAT/2021/001",
             self._hash("123456")),
            ("STU002", "John",
             "john@student.com",
             "MAT/2021/002",
             self._hash("123456")),
            ("STU003", "Mary",
             "mary@student.com",
             "MAT/2021/003",
             self._hash("123456")),
            ("STU004", "Paul",
             "paul@student.com",
             "MAT/2021/004",
             self._hash("123456")),
            ("STU005", "Linda",
             "linda@student.com",
             "MAT/2021/005",
             self._hash("123456")),
        ]

        sample_courses = [
            ("CSC306", "Database Management Systems",
             "Computer Science Dept, ABUAD",
             "10:00", "12:00", "Monday", "L002"),
            ("CSC308", "Software Engineering",
             "Computer Science Dept, ABUAD",
             "08:00", "10:00", "Tuesday", "L001"),
            ("CSC318", "Artificial Intelligence",
             "Computer Science Dept, ABUAD",
             "12:00", "14:00", "Wednesday", "L003"),
            ("CSC320", "Computer Networks",
             "Computer Science Dept, ABUAD",
             "14:00", "16:00", "Thursday", "L004"),
            ("CSC322", "Operating Systems",
             "Computer Science Dept, ABUAD",
             "08:00", "10:00", "Friday", "L005"),
        ]

        sample_enrollments = [
            ("STU001", "CSC308"),
            ("STU001", "CSC306"),
            ("STU001", "CSC318"),
            ("STU002", "CSC308"),
            ("STU002", "CSC320"),
            ("STU002", "CSC322"),
            ("STU003", "CSC306"),
            ("STU003", "CSC318"),
            ("STU003", "CSC320"),
            ("STU004", "CSC308"),
            ("STU004", "CSC306"),
            ("STU004", "CSC322"),
            ("STU005", "CSC318"),
            ("STU005", "CSC320"),
            ("STU005", "CSC322"),
        ]

        cleanup_tables = [
            "attendance",
            "feedback",
            "voice_samples",
            "location_log",
            "system_control",
            "sentiment_feedback",
            "access_log",
            "reset_tokens",
            "login_log",
            "email_verifications",
            "voice_templates",
            "qr_sessions",
            "evaluation_results",
            "data_splits",
        ]

        for table in cleanup_tables:
            cursor.execute(f"DELETE FROM {table}")

        cursor.execute(
            "DELETE FROM enrollments"
        )
        cursor.execute(
            "DELETE FROM students "
            "WHERE student_id NOT IN (?, ?, ?, ?, ?)"
        , [s[0] for s in sample_students])
        cursor.execute(
            "DELETE FROM lecturers "
            "WHERE lecturer_id NOT IN (?, ?, ?, ?, ?)"
        , [l[0] for l in sample_lecturers])
        cursor.execute(
            "DELETE FROM courses "
            "WHERE course_code NOT IN (?, ?, ?, ?, ?)"
        , [c[0] for c in sample_courses])

        for lecturer_id, name, email, password in sample_lecturers:
            cursor.execute("""
                INSERT OR REPLACE INTO lecturers
                (lecturer_id, name, email, password,
                 role, first_login, created_at)
                VALUES (?, ?, ?, ?, 'lecturer', 1, ?)
            """, (lecturer_id, name, email, password, now))

        for student_id, name, email, matric, password in sample_students:
            cursor.execute("""
                INSERT OR REPLACE INTO students
                (student_id, name, email,
                 matric_number, password,
                 role, registered_at)
                VALUES (?, ?, ?, ?, ?, 'student', ?)
            """, (student_id, name, email, matric, password, now))

        for course_code, course_title, venue, start_time, end_time, day, lecturer_id in sample_courses:
            cursor.execute("""
                INSERT OR REPLACE INTO courses
                (course_code, course_title, venue,
                 start_time, end_time, day, lecturer_id,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (course_code, course_title, venue,
                   start_time, end_time, day,
                   lecturer_id, now, now))

        cursor.execute(
            "SELECT course_id, course_code FROM courses"
        )
        course_map = {
            row[1]: row[0]
            for row in cursor.fetchall()
        }

        for student_id, course_code in sample_enrollments:
            course_id = course_map.get(course_code)
            if course_id:
                cursor.execute("""
                    INSERT OR IGNORE INTO enrollments
                    (student_id, course_id, joined_at)
                    VALUES (?, ?, ?)
                """, (student_id, course_id, now))

        conn.commit()
        conn.close()
        return True, "System reset to seeded sample data"
    
    def register_student(self, name, email,
                          matric_number, password):
        """
        Register a new student account.

        Validates:
        - Email format and uniqueness
        - Matric number format: YY/SCI01/NNN
          e.g. 22/SCI01/114
        - Password strength (min 4 rules of 5)
        - No duplicate email or matric number

        Returns (success, message, student_id)
        """
        import re

        # ── Validate name
        name = name.strip()
        if len(name) < 2:
            return (False,
                    "Full name must be at least "
                    "2 characters", None)

        # ── Validate email format
        email   = email.strip().lower()
        pattern = (r'^[a-zA-Z0-9._%+\-]+'
                   r'@[a-zA-Z0-9.\-]+'
                   r'\.[a-zA-Z]{2,}$')
        if not re.match(pattern, email):
            return (False,
                    "Invalid email format. "
                    "Use: name@gmail.com",
                    None)

        # ── Validate matric number format
        matric  = matric_number.strip().upper()
        mat_pat = r'^\d{2}/SCI01/\d{3}$'
        if not re.match(mat_pat, matric):
            return (False,
                    "Invalid matric format. "
                    "Use: 22/SCI01/114 "
                    "(YY/SCI01/3digits)",
                    None)

        # ── Validate password strength
        is_strong, msg, score = (
            self.check_password_strength(password))
        if not is_strong:
            return False, msg, None

        conn   = self._get_conn()
        cursor = conn.cursor()

        # ── Check email uniqueness
        cursor.execute("""
            SELECT email FROM students
            WHERE email = ?
        """, (email,))
        if cursor.fetchone():
            conn.close()
            return (False,
                    "Email already registered. "
                    "Please use a different email.",
                    None)

        cursor.execute("""
            SELECT email FROM lecturers
            WHERE email = ?
        """, (email,))
        if cursor.fetchone():
            conn.close()
            return (False,
                    "Email already in use.",
                    None)

        # ── Check matric uniqueness
        cursor.execute("""
            SELECT matric_number FROM students
            WHERE matric_number = ?
        """, (matric,))
        if cursor.fetchone():
            conn.close()
            return (False,
                    "Matric number already registered.",
                    None)

        # ── Generate student_id from matric
        # Format: STU_22SCI01114
        clean_id   = matric.replace("/", "")
        student_id = f"STU_{clean_id}"
        pw_hash    = self._hash(password)
        now        = datetime.now().isoformat()

        try:
            cursor.execute("""
                INSERT INTO students
                (student_id, name, email,
                 matric_number, password,
                 role, registered_at)
                VALUES (?, ?, ?, ?, ?,
                        'student', ?)
            """, (student_id, name, email,
                   matric, pw_hash, now))
            conn.commit()
            conn.close()

            return (True,
                    "Account created successfully!",
                    student_id)

        except sqlite3.IntegrityError as e:
            conn.close()
            return (False,
                    "Registration failed — "
                    "please try again.",
                    None)
    
    def store_signup_verification(self,
                                   email,
                                   code,
                                   expires_at,
                                   pending_data):
        """
        Store signup verification code before
        account is created. Account only created
        after email is confirmed.
        """
        import json
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS
            signup_verifications (
                id           INTEGER PRIMARY KEY
                             AUTOINCREMENT,
                email        TEXT NOT NULL,
                code         TEXT NOT NULL,
                expires_at   TEXT NOT NULL,
                pending_data TEXT NOT NULL,
                verified     INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL
            )
        """)

        # Remove old pending codes for same email
        cursor.execute("""
            DELETE FROM signup_verifications
            WHERE email = ?
        """, (email.lower(),))

        cursor.execute("""
            INSERT INTO signup_verifications
            (email, code, expires_at,
             pending_data, verified, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (email.lower(), str(code),
               str(expires_at),
               json.dumps(pending_data),
               datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def verify_signup_code(self, email, code):
        """
        Verify the signup email code.
        Returns (success, message, pending_data).
        On success: returns stored registration data.
        """
        import json
        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT code, expires_at,
                       pending_data, verified
                FROM signup_verifications
                WHERE email = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (email.lower(),))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return (False,
                        "No verification found. "
                        "Please sign up again.",
                        None)

            if int(row[3]) == 1:
                conn.close()
                return (False,
                        "Code already used.",
                        None)

            if str(row[0]).strip() != \
                    str(code).strip():
                conn.close()
                return (False,
                        "Incorrect code. "
                        "Please try again.",
                        None)

            try:
                exp = datetime.fromisoformat(
                    str(row[1]))
                if datetime.now() > exp:
                    conn.close()
                    return (False,
                            "Code has expired. "
                            "Please sign up again.",
                            None)
            except Exception:
                pass

            # Mark as verified
            cursor.execute("""
                UPDATE signup_verifications
                SET verified = 1
                WHERE email  = ?
            """, (email.lower(),))
            conn.commit()

            pending = json.loads(row[2])
            conn.close()
            return True, "Email verified!", pending

        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            return False, str(e), None

    def student_has_voice_enrolled(self,
                                    student_id):
        """
        Check if student has recorded voice samples.
        Returns True if voice template exists.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM voice_samples
            WHERE student_id = ?
        """, (student_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0

    def save_voice_enrollment(self, student_id,
                               features_list,
                               sample_count):
        """
        Save voice enrollment data for a student.
        Stores MFCC features from multiple recordings.

        features_list: list of numpy arrays (MFCCs)
        sample_count : number of recordings made
        """
        import json
        import numpy as np

        conn   = self._get_conn()
        cursor = conn.cursor()

        # Create voice_templates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_templates (
                id              INTEGER PRIMARY KEY
                                AUTOINCREMENT,
                student_id      TEXT NOT NULL UNIQUE,
                features_json   TEXT NOT NULL,
                sample_count    INTEGER NOT NULL,
                enrolled_at     TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
        """)

        # Serialize features to JSON
        serialized = json.dumps([
            f.tolist() for f in features_list
        ])

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO voice_templates
            (student_id, features_json,
             sample_count, enrolled_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (student_id, serialized,
               sample_count, now, now))

        # Also log in voice_samples table
        cursor.execute("""
            DELETE FROM voice_samples
            WHERE student_id = ?
              AND filename LIKE 'enrollment_%'
        """, (student_id,))

        for i in range(sample_count):
            cursor.execute("""
                INSERT INTO voice_samples
                (student_id, filename, file_path,
                 sample_rate, format, recorded_at)
                VALUES (?, ?, ?, 16000, 'wav', ?)
            """, (student_id,
                   f"enrollment_{i+1}.wav",
                   f"enrollment/{student_id}/"
                   f"sample_{i+1}.wav",
                   now))

        # Update voice_template flag on students
        cursor.execute("""
            UPDATE students
            SET registered_at = registered_at
            WHERE student_id = ?
        """, (student_id,))

        conn.commit()
        conn.close()
        return True

    def get_voice_template(self, student_id):
        """
        Retrieve voice template features for a student.
        Returns list of numpy arrays or None.
        """
        import json
        import numpy as np

        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                voice_templates (
                    id              INTEGER PRIMARY KEY
                                    AUTOINCREMENT,
                    student_id      TEXT NOT NULL UNIQUE,
                    features_json   TEXT NOT NULL,
                    sample_count    INTEGER NOT NULL,
                    enrolled_at     TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
            """)

            cursor.execute("""
                SELECT features_json, sample_count,
                       enrolled_at
                FROM voice_templates
                WHERE student_id = ?
            """, (student_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            features = [
                np.array(f)
                for f in json.loads(row[0])
            ]
            return {
                "features"    : features,
                "sample_count": row[1],
                "enrolled_at" : row[2]
            }

        except Exception:
            conn.close()
            return None

    def get_enrollment_status(self, student_id):
        """
        Check voice enrollment status.
        Returns dict with enrolled bool + details.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                voice_templates (
                    id              INTEGER PRIMARY KEY
                                    AUTOINCREMENT,
                    student_id      TEXT NOT NULL UNIQUE,
                    features_json   TEXT NOT NULL,
                    sample_count    INTEGER NOT NULL,
                    enrolled_at     TEXT NOT NULL,
                    updated_at      TEXT NOT NULL
                )
            """)

            cursor.execute("""
                SELECT sample_count, enrolled_at
                FROM voice_templates
                WHERE student_id = ?
            """, (student_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "enrolled"    : True,
                    "sample_count": row[0],
                    "enrolled_at" : row[1]
                }
            return {
                "enrolled"    : False,
                "sample_count": 0,
                "enrolled_at" : None
            }
        except Exception:
            conn.close()
            return {"enrolled": False,
                    "sample_count": 0,
                    "enrolled_at": None}

    # ─────────────────────────────────────────
    # QR CODE METHODS
    # ─────────────────────────────────────────
    def create_qr_session(self, course_code,
                           lecturer_id,
                           expires_seconds=30):
        """
        Create a dynamic QR session token.
        Stores all fields explicitly as strings.
        """
        import secrets as sec
        import hmac
        import hashlib
        import json

        conn   = self._get_conn()
        cursor = conn.cursor()

        # Ensure table exists with correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qr_sessions (
                id           INTEGER PRIMARY KEY
                             AUTOINCREMENT,
                course_code  TEXT NOT NULL,
                lecturer_id  TEXT NOT NULL,
                token        TEXT NOT NULL UNIQUE,
                session_id   TEXT NOT NULL,
                payload_json TEXT,
                created_at   TEXT NOT NULL,
                expires_at   TEXT NOT NULL,
                used_by      TEXT DEFAULT NULL
            )
        """)

        # Clean up expired tokens first
        now_str = datetime.now().isoformat()
        cursor.execute("""
            DELETE FROM qr_sessions
            WHERE expires_at < ?
               OR expires_at IS NULL
               OR expires_at = ''
        """, (now_str,))

        # Generate token components
        session_id = sec.token_hex(8).upper()
        raw_token  = sec.token_urlsafe(24)
        now        = datetime.now()
        expires    = now + timedelta(
            seconds=expires_seconds)

        # Explicitly convert to ISO strings
        now_iso    = now.isoformat()
        expires_iso= expires.isoformat()
        timestamp  = int(now.timestamp())

        # Verify they are strings
        assert isinstance(now_iso, str), \
            "created_at must be string"
        assert isinstance(expires_iso, str), \
            "expires_at must be string"

        # Build payload
        payload = {
            "sessionId"  : session_id,
            "courseCode" : course_code.upper(),
            "timestamp"  : timestamp,
            "token"      : raw_token
        }

        # HMAC signature
        secret    = (f"{course_code}"
                     f"{session_id}"
                     f"{timestamp}")
        signature = hmac.new(
            secret.encode(),
            raw_token.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
        payload["signature"] = signature

        payload_str = json.dumps(
            payload, separators=(',', ':'))

        cursor.execute("""
            INSERT INTO qr_sessions
            (course_code, lecturer_id, token,
             session_id, payload_json,
             created_at, expires_at, used_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """, (
            course_code.upper(),
            str(lecturer_id),
            str(raw_token),
            str(session_id),
            str(payload_str),
            str(now_iso),
            str(expires_iso)
        ))

        conn.commit()

        # Verify the insert immediately
        cursor.execute("""
            SELECT token, expires_at, created_at
            FROM qr_sessions
            WHERE token = ?
        """, (raw_token,))
        verify_row = cursor.fetchone()
        conn.close()

        if verify_row:
            print(f"  ✅ QR created: "
                  f"expires={verify_row[1]}")
        else:
            print(f"  ❌ QR insert failed!")

        return {
            "token"         : raw_token,
            "token_json"    : payload_str,
            "session_id"    : session_id,
            "payload"       : payload,
            "expires_at"    : expires_iso,
            "expires_in"    : expires_seconds,
            "course_code"   : course_code.upper(),
            "timestamp"     : timestamp
        }

    def verify_qr_token(self, token,
                         course_code,
                         student_id):
        """
        Verify a QR token for attendance.
        Handles all edge cases for expires_at field.
        """
        import json as json_mod

        # Extract raw token from JSON payload if needed
        raw_token = str(token).strip()
        if raw_token.startswith('{'):
            try:
                payload   = json_mod.loads(raw_token)
                raw_token = str(payload.get(
                    "token", raw_token))
            except Exception:
                pass

        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                qr_sessions (
                    id           INTEGER PRIMARY KEY
                                 AUTOINCREMENT,
                    course_code  TEXT NOT NULL,
                    lecturer_id  TEXT NOT NULL,
                    token        TEXT NOT NULL UNIQUE,
                    session_id   TEXT NOT NULL,
                    payload_json TEXT,
                    created_at   TEXT NOT NULL,
                    expires_at   TEXT NOT NULL,
                    used_by      TEXT DEFAULT NULL
                )
            """)

            cursor.execute("""
                SELECT id, course_code, token,
                       expires_at, used_by
                FROM qr_sessions
                WHERE token = ?
                  AND course_code = ?
                  AND (used_by IS NULL
                       OR used_by = '')
                ORDER BY id DESC
                LIMIT 1
            """, (raw_token,
                   course_code.upper()))
            row = cursor.fetchone()

            if not row:
                # Try without course_code filter
                # (in case course_code mismatch)
                cursor.execute("""
                    SELECT id, course_code, token,
                           expires_at, used_by
                    FROM qr_sessions
                    WHERE token = ?
                      AND (used_by IS NULL
                           OR used_by = '')
                    ORDER BY id DESC
                    LIMIT 1
                """, (raw_token,))
                row = cursor.fetchone()

            if not row:
                conn.close()
                return (False,
                        "Invalid QR code. "
                        "Please get the current "
                        "token from your lecturer.")

            row_id     = row[0]
            expires_raw = row[3]
            used_by    = row[4]

            print(f"  🔍 QR verify: "
                  f"token={raw_token[:12]}... "
                  f"expires_raw={expires_raw!r} "
                  f"type={type(expires_raw).__name__}")

            # Already used check
            if used_by and str(used_by).strip():
                conn.close()
                return (False,
                        "This QR token has already "
                        "been used.")

            # Validate expires_at field
            if expires_raw is None or \
                    expires_raw == '':
                conn.close()
                return (False,
                        "QR session data is invalid. "
                        "Please ask lecturer to "
                        "regenerate the QR code.")

            # Convert to string if somehow not
            expires_str = str(expires_raw).strip()

            if not expires_str:
                conn.close()
                return (False,
                        "QR session data is corrupted. "
                        "Regenerate QR code.")

            # Parse expiry time
            try:
                expires = datetime.fromisoformat(
                    expires_str)
            except (ValueError, TypeError) as e:
                print(f"  ❌ fromisoformat failed: "
                      f"{e!r} on {expires_str!r}")
                conn.close()
                return (False,
                        f"QR timestamp format error. "
                        f"Please regenerate QR code.")

            # Expiry check
            now = datetime.now()
            if now > expires:
                diff = int((now - expires).total_seconds())
                conn.close()
                return (False,
                        f"QR code expired {diff}s ago. "
                        f"Please scan the current "
                        f"QR code shown by your lecturer.")

            # All checks passed — mark as used
            cursor.execute("""
                UPDATE qr_sessions
                SET used_by = ?
                WHERE id = ?
            """, (str(student_id), row_id))
            conn.commit()
            conn.close()

            print(f"  ✅ QR verified for "
                  f"student {student_id}")
            return True, "QR verified successfully"

        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                conn.close()
            except Exception:
                pass
            return False, f"Verification error: {str(e)}"

    def get_active_qr(self, course_code):
        """Get current active QR data."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                qr_sessions (
                    id          INTEGER PRIMARY KEY
                                AUTOINCREMENT,
                    course_code TEXT NOT NULL,
                    lecturer_id TEXT NOT NULL,
                    token       TEXT NOT NULL UNIQUE,
                    session_id  TEXT NOT NULL,
                    payload_json TEXT,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    used_by     TEXT DEFAULT NULL
                )
            """)
            cursor.execute("""
                SELECT token, expires_at,
                       session_id, payload_json
                FROM qr_sessions
                WHERE course_code = ?
                  AND expires_at  > ?
                  AND used_by     IS NULL
                ORDER BY created_at DESC
                LIMIT 1
            """, (course_code.upper(),
                   datetime.now().isoformat()))
            row = cursor.fetchone()
            conn.close()
            if row:
                # Validate expires_at before returning
                expires_str = row[1]
                if expires_str and \
                        isinstance(expires_str, str):
                    try:
                        exp = datetime.fromisoformat(
                            expires_str)
                        if datetime.now() > exp:
                            conn.close()
                            return None
                    except (ValueError, TypeError):
                        conn.close()
                        return None
                return {
                    "token"     : row[0],
                    "expires_at": row[1],
                    "session_id": row[2],
                    "token_json": row[3]
                }
        except Exception:
            conn.close()
        return None

    # ─────────────────────────────────────────
    # FIREBASE
    # ─────────────────────────────────────────
    def _init_firebase(self):
        if not FIREBASE_ENABLED:
            print("  ℹ️  Firebase: Disabled (SQLite only)")
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, db
            if not firebase_admin._apps:
                cred = credentials.Certificate(
                    FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_DB_URL
                })
            self.firebase_db = db
            print("  ✅ Firebase: Connected")
        except ImportError:
            print("  ⚠️  Run: pip install firebase-admin")
        except Exception as e:
            print(f"  ⚠️  Firebase: {e}")

    def _sync_firebase(self, collection,
                        record_id, data):
        """Sync record to Firebase in background."""
        try:
            ref = self.firebase_db.reference(
                f"/{collection}/{record_id}")
            ref.set(data)
            conn   = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {collection} "
                f"SET synced_firebase = 1 "
                f"WHERE id = ?", (record_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ─────────────────────────────────────────
    # AUTHENTICATION
    # ─────────────────────────────────────────
    def login(self, email, password):
        """
        Unified login for students and lecturers.
        Returns user dict with role, or None.

        Role-based routing:
        - 'lecturer' → lecturer dashboard
        - 'student'  → student dashboard
        """
        pw_hash = self._hash(password)
        conn    = self._get_conn()
        cursor  = conn.cursor()

        # Check lecturers
        cursor.execute("""
            SELECT lecturer_id, name, email,
                   role, first_login
            FROM lecturers
            WHERE email = ? AND password = ?
        """, (email, pw_hash))
        row = cursor.fetchone()

        if row:
            conn.close()
            return {
                "id"         : row[0],
                "name"       : row[1],
                "email"      : row[2],
                "role"       : row[3],
                "first_login": bool(row[4])
            }

        # Check students
        cursor.execute("""
            SELECT student_id, name, email,
                   matric_number, role
            FROM students
            WHERE email = ? AND password = ?
        """, (email, pw_hash))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id"           : row[0],
                "name"         : row[1],
                "email"        : row[2],
                "matric_number": row[3],
                "role"         : row[4]
            }

        return None

    # ─────────────────────────────────────────
    # PASSWORD RESET METHODS
    # ─────────────────────────────────────────
    def store_reset_token(self, email, token,
                           expires_at):
        """
        Store password reset token for a user.
        Works for both students and lecturers.
        Token expires after 1 hour.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        # Create reset_tokens table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reset_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT NOT NULL,
                token      TEXT NOT NULL UNIQUE,
                role       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        # Delete any existing token for this email
        cursor.execute("""
            DELETE FROM reset_tokens WHERE email = ?
        """, (email,))

        # Determine role
        role = self._get_user_role(email)

        cursor.execute("""
            INSERT INTO reset_tokens
            (email, token, role, expires_at,
             used, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
        """, (email, token, role, expires_at,
               datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def verify_reset_token(self, token):
        """
        Verify a password reset token.
        Returns user dict if valid, None if invalid/expired.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reset_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT NOT NULL,
                token      TEXT NOT NULL UNIQUE,
                role       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            SELECT * FROM reset_tokens
            WHERE token = ? AND used = 0
        """, (token,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Check expiry
        expires = datetime.fromisoformat(row[4])
        if datetime.now() > expires:
            return None

        return {
            "email"     : row[1],
            "token"     : row[2],
            "role"      : row[3],
            "expires_at": row[4]
        }

    def reset_password(self, token, new_password):
        """
        Reset password using valid token.
        Updates password for student or lecturer.
        Marks token as used after reset.
        Returns (success, message).
        """
        token_data = self.verify_reset_token(token)

        if not token_data:
            return False, "Invalid or expired reset link"

        email    = token_data["email"]
        role     = token_data["role"]
        pw_hash  = self._hash(new_password)
        conn     = self._get_conn()
        cursor   = conn.cursor()

        if role == "lecturer":
            cursor.execute("""
                UPDATE lecturers
                SET password = ?, first_login = 0
                WHERE email = ?
            """, (pw_hash, email))
        else:
            cursor.execute("""
                UPDATE students
                SET password = ?
                WHERE email = ?
            """, (pw_hash, email))

        # Mark token as used
        cursor.execute("""
            UPDATE reset_tokens
            SET used = 1
            WHERE token = ?
        """, (token,))

        conn.commit()
        conn.close()
        return True, "Password reset successfully"

    def update_password_direct(self, email,
                                new_password,
                                mark_first_login_done=True):
        """
        Directly update password.
        Used for first-login forced password change.
        Marks first_login = 0 after update.
        """
        pw_hash = self._hash(new_password)
        role    = self._get_user_role(email)
        conn    = self._get_conn()
        cursor  = conn.cursor()

        if role == "lecturer":
            cursor.execute("""
                UPDATE lecturers
                SET password    = ?,
                    first_login = ?
                WHERE email = ?
            """, (pw_hash,
                   0 if mark_first_login_done else 1,
                   email))
        else:
            cursor.execute("""
                UPDATE students
                SET password = ?
                WHERE email  = ?
            """, (pw_hash, email))

        conn.commit()
        conn.close()
        return True

    def check_password_strength(self, password):
        """
        Check password strength server-side.
        Returns (is_strong, message, score 0-5).

        Rules:
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character
        """
        import re
        score    = 0
        issues   = []

        if len(password) >= 8:
            score += 1
        else:
            issues.append(
                "At least 8 characters required")

        if re.search(r'[A-Z]', password):
            score += 1
        else:
            issues.append(
                "At least one uppercase letter")

        if re.search(r'[a-z]', password):
            score += 1
        else:
            issues.append(
                "At least one lowercase letter")

        if re.search(r'[0-9]', password):
            score += 1
        else:
            issues.append(
                "At least one number")

        if re.search(r'[^A-Za-z0-9]', password):
            score += 1
        else:
            issues.append(
                "At least one special character "
                "(!@#$%^&*...)")

        is_strong = score >= 4
        message   = (
            "Password is strong" if is_strong
            else "Weak password: " +
                 ", ".join(issues))

        return is_strong, message, score

    def email_exists(self, email):
        """
        Check if email exists in either
        students or lecturers table.
        Returns (exists, role) tuple.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT email FROM lecturers WHERE email=?",
            (email,))
        if cursor.fetchone():
            conn.close()
            return True, "lecturer"

        cursor.execute(
            "SELECT email FROM students WHERE email=?",
            (email,))
        if cursor.fetchone():
            conn.close()
            return True, "student"

        conn.close()
        return False, None

    def _get_user_role(self, email):
        """Get role for an email address."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT email FROM lecturers WHERE email=?",
            (email,))
        if cursor.fetchone():
            conn.close()
            return "lecturer"
        conn.close()
        return "student"

    def cleanup_expired_tokens(self):
        """Remove expired reset tokens."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                DELETE FROM reset_tokens
                WHERE expires_at < ? OR used = 1
            """, (datetime.now().isoformat(),))
            conn.commit()
        except Exception:
            pass
        conn.close()

    # ─────────────────────────────────────────
    # USER PROFILE METHODS
    # ─────────────────────────────────────────
    def get_user_profile(self, user_id, role):
        """
        Get full profile for student or lecturer.
        Returns profile dict with all display fields.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        if role == "lecturer":
            cursor.execute("""
                SELECT lecturer_id, name, email,
                       role, first_login, created_at
                FROM lecturers
                WHERE lecturer_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "id"          : row[0],
                "name"        : row[1],
                "email"       : row[2],
                "role"        : row[3],
                "first_login" : bool(row[4]),
                "created_at"  : row[5],
                "profile_pic" : self._get_profile_pic(
                    user_id),
                "last_login"  : self._get_last_login(
                    user_id, role)
            }
        else:
            cursor.execute("""
                SELECT student_id, name, email,
                       matric_number, role,
                       registered_at
                FROM students
                WHERE student_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "id"           : row[0],
                "name"         : row[1],
                "email"        : row[2],
                "matric_number": row[3],
                "role"         : row[4],
                "created_at"   : row[5],
                "profile_pic"  : self._get_profile_pic(
                    user_id),
                "last_login"   : self._get_last_login(
                    user_id, role)
            }

    def update_profile_name(self, user_id,
                             role, new_name):
        """
        Update display name for student or lecturer.
        Returns (success, message).
        """
        if not new_name or len(new_name.strip()) < 2:
            return False, "Name must be at least 2 characters"

        new_name = new_name.strip()
        conn     = self._get_conn()
        cursor   = conn.cursor()

        if role == "lecturer":
            cursor.execute("""
                UPDATE lecturers SET name = ?
                WHERE lecturer_id = ?
            """, (new_name, user_id))
        else:
            cursor.execute("""
                UPDATE students SET name = ?
                WHERE student_id = ?
            """, (new_name, user_id))

        conn.commit()
        conn.close()
        return True, "Name updated successfully"
    
    def store_email_verification_code(self,
                                       user_id,
                                       role,
                                       new_email,
                                       code,
                                       expires_at):
        """
        Store email verification code.
        Used for email update verification flow.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS
            email_verifications (
                id          INTEGER PRIMARY KEY
                            AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                new_email   TEXT NOT NULL,
                code        TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                verified    INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL
            )
        """)

        # Delete any existing pending code for user
        cursor.execute("""
            DELETE FROM email_verifications
            WHERE user_id = ? AND verified = 0
        """, (user_id,))

        cursor.execute("""
            INSERT INTO email_verifications
            (user_id, role, new_email, code,
             expires_at, verified, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
        """, (user_id, role, new_email.lower().strip(),
               code, expires_at,
               datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def verify_email_code(self, user_id, code):
        """
        Verify email update code.
        Returns (success, message, new_email, role).
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                email_verifications (
                    id          INTEGER PRIMARY KEY
                                AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    new_email   TEXT NOT NULL,
                    code        TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    verified    INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL
                )
            """)

            cursor.execute("""
                SELECT * FROM email_verifications
                WHERE user_id = ?
                  AND code    = ?
                  AND verified = 0
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id, code))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return (False,
                        "Invalid verification code",
                        None, None)

            expires = datetime.fromisoformat(row[5])
            if datetime.now() > expires:
                conn.close()
                return (False,
                        "Code has expired. "
                        "Please request a new one.",
                        None, None)

            new_email = row[3]
            role      = row[2]

            # Mark as verified
            cursor.execute("""
                UPDATE email_verifications
                SET verified = 1
                WHERE id = ?
            """, (row[0],))
            conn.commit()
            conn.close()

            return True, "Code verified", new_email, role

        except Exception as e:
            conn.close()
            return False, str(e), None, None

    def store_email_verification_code(self,
                                       user_id,
                                       role,
                                       new_email,
                                       code,
                                       expires_at):
        """
        Store email verification code.
        Used for email update verification flow.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS
            email_verifications (
                id          INTEGER PRIMARY KEY
                            AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                new_email   TEXT NOT NULL,
                code        TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                verified    INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL
            )
        """)

        # Delete any existing pending code for user
        cursor.execute("""
            DELETE FROM email_verifications
            WHERE user_id = ? AND verified = 0
        """, (user_id,))

        cursor.execute("""
            INSERT INTO email_verifications
            (user_id, role, new_email, code,
             expires_at, verified, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?)
        """, (user_id, role, new_email.lower().strip(),
               code, expires_at,
               datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def verify_email_code(self, user_id, code):
        """
        Verify email update code.
        Returns (success, message, new_email, role).
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS
                email_verifications (
                    id          INTEGER PRIMARY KEY
                                AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    new_email   TEXT NOT NULL,
                    code        TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    verified    INTEGER DEFAULT 0,
                    created_at  TEXT NOT NULL
                )
            """)

            cursor.execute("""
                SELECT * FROM email_verifications
                WHERE user_id = ?
                  AND code    = ?
                  AND verified = 0
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id, code))
            row = cursor.fetchone()

            if not row:
                conn.close()
                return (False,
                        "Invalid verification code",
                        None, None)

            expires = datetime.fromisoformat(row[5])
            if datetime.now() > expires:
                conn.close()
                return (False,
                        "Code has expired. "
                        "Please request a new one.",
                        None, None)

            new_email = row[3]
            role      = row[2]

            # Mark as verified
            cursor.execute("""
                UPDATE email_verifications
                SET verified = 1
                WHERE id = ?
            """, (row[0],))
            conn.commit()
            conn.close()

            return True, "Code verified", new_email, role

        except Exception as e:
            conn.close()
            return False, str(e), None, None

    def update_email(self, user_id, role, new_email):
        """
        Update email after verification.
        Checks uniqueness before updating.
        """
        import re
        new_email = new_email.strip().lower()

        # Format validation
        pattern = (r'^[a-zA-Z0-9._%+\-]+'
                   r'@[a-zA-Z0-9.\-]+'
                   r'\.[a-zA-Z]{2,}$')
        if not re.match(pattern, new_email):
            return False, "Invalid email format"

        conn   = self._get_conn()
        cursor = conn.cursor()

        # Check uniqueness
        cursor.execute(
            "SELECT email FROM lecturers "
            "WHERE email = ?", (new_email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already in use"

        cursor.execute(
            "SELECT email FROM students "
            "WHERE email = ?", (new_email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already in use"

        if role == "lecturer":
            cursor.execute("""
                UPDATE lecturers SET email = ?
                WHERE lecturer_id = ?
            """, (new_email, user_id))
        else:
            cursor.execute("""
                UPDATE students SET email = ?
                WHERE student_id = ?
            """, (new_email, user_id))

        conn.commit()
        conn.close()
        return True, "Email updated successfully"

    def update_course_details(self, course_code,
                               venue=None,
                               start_time=None,
                               end_time=None,
                               lecturer_id=None):
        """
        Update course venue and/or time.
        If lecturer_id provided, verifies ownership.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        if lecturer_id:
            cursor.execute("""
                SELECT lecturer_id FROM courses
                WHERE course_code = ?
            """, (course_code.upper(),))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False, "Course not found"
            if row[0] != lecturer_id:
                conn.close()
                return (False,
                        "Access denied — not your course")

        updates = []
        values  = []

        if venue:
            updates.append("venue = ?")
            values.append(venue)
        if start_time:
            updates.append("start_time = ?")
            values.append(start_time)
        if end_time:
            updates.append("end_time = ?")
            values.append(end_time)

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(course_code.upper())

        cursor.execute(
            f"UPDATE courses SET "
            f"{', '.join(updates)} "
            f"WHERE course_code = ?",
            values)
        conn.commit()
        conn.close()
        return True, "Course updated"

    def remove_course_enrollment(self, student_id,
                                  course_code):
        """Remove a student's enrollment from a course."""
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT course_id FROM courses
            WHERE course_code = ?
        """, (course_code.upper(),))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return False, "Course not found"

        course_id = row[0]
        cursor.execute("""
            DELETE FROM enrollments
            WHERE student_id = ? AND course_id = ?
        """, (student_id, course_id))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        if affected:
            return True, "Course removed successfully"
        return False, "You are not enrolled in this course"

    def delete_course(self, course_code, lecturer_id):
        """
        Delete a course owned by a lecturer.
        Also removes all enrollments for that course.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        # Verify ownership
        cursor.execute("""
            SELECT course_id, lecturer_id
            FROM courses WHERE course_code = ?
        """, (course_code.upper(),))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return False, "Course not found"

        if row[1] != lecturer_id:
            conn.close()
            return (False,
                    "Access denied — you do not "
                    "own this course")

        course_id = row[0]

        # Remove enrollments first
        cursor.execute("""
            DELETE FROM enrollments
            WHERE course_id = ?
        """, (course_id,))

        # Remove system control records
        cursor.execute("""
            DELETE FROM system_control
            WHERE course_code = ?
        """, (course_code.upper(),))

        # Remove course
        cursor.execute("""
            DELETE FROM courses
            WHERE course_id = ?
        """, (course_id,))

        conn.commit()
        conn.close()
        return True, f"Course {course_code} deleted"

    def get_student_course_attendance_counts(self,
                                              student_id):
        """
        Get attendance count per course for a student.
        Returns dict: {course_code: count}
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT course_code, COUNT(*) as cnt
            FROM attendance
            WHERE student_id = ?
              AND status = 'PRESENT'
            GROUP BY course_code
        """, (student_id,))
        rows = cursor.fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows if r[0]} 

    def remove_profile_picture(self, user_id):
        """Remove profile picture for a user."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id     TEXT PRIMARY KEY,
                    profile_pic TEXT,
                    pic_ext     TEXT DEFAULT 'jpg',
                    updated_at  TEXT NOT NULL
                )
            """)
            cursor.execute("""
                UPDATE user_profiles
                SET profile_pic = NULL,
                    updated_at  = ?
                WHERE user_id = ?
            """, (datetime.now().isoformat(), user_id))
            conn.commit()
            conn.close()
            return True, "Profile picture removed"
        except Exception as e:
            conn.close()
            return False, str(e)

    def update_matric_number(self, student_id,
                              matric_number):
        """
        Update matric number for a student.
        Format: 22/SCI01/*** where *** = 3-digit number
        e.g. 22/SCI01/114
        Validates format before saving.
        """
        import re

        matric = matric_number.strip().upper()

        # Validate format: YY/XXXNN/NNN
        # e.g. 22/SCI01/114
        pattern = r'^\d{2}/[A-Z]{2,4}\d{1,3}/\d{3}$'
        if not re.match(pattern, matric):
            return (False,
                    "Invalid format. Use: "
                    "22/SCI01/114 "
                    "(YY/DEPT+NUM/3digits)")

        conn   = self._get_conn()
        cursor = conn.cursor()

        # Check not already used by another student
        cursor.execute("""
            SELECT student_id FROM students
            WHERE matric_number = ?
              AND student_id != ?
        """, (matric, student_id))
        if cursor.fetchone():
            conn.close()
            return (False,
                    "Matric number already in use "
                    "by another student")

        cursor.execute("""
            UPDATE students
            SET matric_number = ?
            WHERE student_id = ?
        """, (matric, student_id))
        conn.commit()
        conn.close()
        return True, "Matric number updated"

    def save_profile_picture(self, user_id,
                              pic_data, pic_ext="jpg"):
        """
        Save profile picture as base64 in database.
        Stores in user_profiles table.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id     TEXT PRIMARY KEY,
                profile_pic TEXT,
                pic_ext     TEXT DEFAULT 'jpg',
                updated_at  TEXT NOT NULL
            )
        """)

        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles
            (user_id, profile_pic, pic_ext, updated_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, pic_data, pic_ext,
               datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    def _get_profile_pic(self, user_id):
        """Get profile picture data for a user."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id     TEXT PRIMARY KEY,
                    profile_pic TEXT,
                    pic_ext     TEXT DEFAULT 'jpg',
                    updated_at  TEXT NOT NULL
                )
            """)
            cursor.execute("""
                SELECT profile_pic, pic_ext
                FROM user_profiles
                WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                return {
                    "data": row[0],
                    "ext" : row[1]
                }
        except Exception:
            conn.close()
        return None

    def log_login(self, user_id, role):
        """
        Log user login timestamp.
        Used for 'Last Login' display on profile.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT NOT NULL,
                role      TEXT NOT NULL,
                logged_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            INSERT INTO login_log
            (user_id, role, logged_at)
            VALUES (?, ?, ?)
        """, (user_id, role,
               datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def _get_last_login(self, user_id, role):
        """Get the last login timestamp for a user."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS login_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    logged_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                SELECT logged_at FROM login_log
                WHERE user_id = ? AND role = ?
                ORDER BY logged_at DESC
                LIMIT 2
            """, (user_id, role))
            rows = cursor.fetchall()
            conn.close()
            # Return second-to-last (last before current)
            if len(rows) >= 2:
                return rows[1][0]
            elif len(rows) == 1:
                return rows[0][0]
        except Exception:
            conn.close()
        return None

    def get_user_course_list(self, user_id, role):
        """
        Get course list for profile display.
        Lecturer: courses they own.
        Student: courses they are enrolled in.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()

        if role == "lecturer":
            cursor.execute("""
                SELECT course_code, course_title,
                       venue, start_time, end_time
                FROM courses
                WHERE lecturer_id = ?
                ORDER BY course_code
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT c.course_code, c.course_title,
                       c.venue, c.start_time, c.end_time
                FROM courses c
                JOIN enrollments e
                    ON c.course_id = e.course_id
                WHERE e.student_id = ?
                ORDER BY c.course_code
            """, (user_id,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "course_code" : r[0],
                "course_title": r[1],
                "venue"       : r[2],
                "start_time"  : r[3],
                "end_time"    : r[4]
            }
            for r in rows
        ]

    def get_user_attendance_stats(self, user_id):
        """
        Get attendance statistics for student profile.
        Returns total, present, rate.
        """
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status='PRESENT'
                    THEN 1 ELSE 0 END) as present
            FROM attendance
            WHERE student_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()

        total   = row[0] or 0
        present = row[1] or 0
        rate    = round(present/total*100, 1) \
                  if total > 0 else 0

        return {
            "total"  : total,
            "present": present,
            "rate"   : rate
        }
    
    # ─────────────────────────────────────────
    # LECTURER METHODS
    # ─────────────────────────────────────────
    def get_lecturer(self, lecturer_id=None,
                      email=None):
        """Get lecturer by ID or email."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        if lecturer_id:
            cursor.execute(
                "SELECT * FROM lecturers "
                "WHERE lecturer_id = ?",
                (lecturer_id,))
        else:
            cursor.execute(
                "SELECT * FROM lecturers "
                "WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        return row

    def verify_lecturer(self, email, password):
        """Verify lecturer credentials."""
        user = self.login(email, password)
        if user and user["role"] == "lecturer":
            return user
        return None

    # ─────────────────────────────────────────
    # STUDENT METHODS
    # ─────────────────────────────────────────
    def get_student(self, student_id=None,
                     email=None):
        """Get student by ID or email."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        if student_id:
            cursor.execute(
                "SELECT * FROM students "
                "WHERE student_id = ?",
                (student_id,))
        else:
            cursor.execute(
                "SELECT * FROM students "
                "WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        return row

    def verify_student(self, email, password):
        """Verify student credentials."""
        user = self.login(email, password)
        if user and user["role"] == "student":
            return user
        return None

    # ─────────────────────────────────────────
    # COURSE METHODS
    # ─────────────────────────────────────────
    def create_course(self, course_code,
                       course_title, venue,
                       start_time, end_time, day,
                       lecturer_id):
        """Create course owned by lecturer."""
        now  = datetime.now().isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO courses
                (course_code, course_title, venue,
                 start_time, end_time, day,
                 lecturer_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (course_code.upper(), course_title,
                   venue, start_time, end_time, day,
                   lecturer_id, now, now))
            conn.commit()
            course_id = cursor.lastrowid
            conn.close()
            return course_id
        except sqlite3.IntegrityError:
            conn.close()
            return None

    def update_course(self, course_code,
                       course_title=None,
                       venue=None,
                       start_time=None,
                       end_time=None,
                       day=None):
        """Update course details."""
        conn    = self._get_conn()
        cursor  = conn.cursor()
        updates = []
        values  = []

        if course_title:
            updates.append("course_title = ?")
            values.append(course_title)
        if venue:
            updates.append("venue = ?")
            values.append(venue)
        if start_time:
            updates.append("start_time = ?")
            values.append(start_time)
        if end_time:
            updates.append("end_time = ?")
            values.append(end_time)
        if day:
            updates.append("day = ?")
            values.append(day)

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(course_code.upper())

        cursor.execute(
            f"UPDATE courses SET "
            f"{', '.join(updates)} "
            f"WHERE course_code = ?",
            values)
        conn.commit()
        conn.close()

    def get_courses(self, lecturer_id=None):
        """Get all courses or by lecturer."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        if lecturer_id:
            cursor.execute("""
                SELECT c.course_id, c.course_code,
                       c.course_title, c.venue,
                       c.start_time, c.end_time,
                       c.day, c.lecturer_id,
                       c.created_at, c.updated_at,
                       l.name as lecturer_name
                FROM courses c
                JOIN lecturers l
                    ON c.lecturer_id = l.lecturer_id
                WHERE c.lecturer_id = ?
                ORDER BY c.course_code
            """, (lecturer_id,))
        else:
            cursor.execute("""
                SELECT c.course_id, c.course_code,
                       c.course_title, c.venue,
                       c.start_time, c.end_time,
                       c.day, c.lecturer_id,
                       c.created_at, c.updated_at,
                       l.name as lecturer_name
                FROM courses c
                JOIN lecturers l
                    ON c.lecturer_id = l.lecturer_id
                ORDER BY c.course_code
            """)
        courses = cursor.fetchall()
        conn.close()
        return courses

    def get_course(self, course_code):
        """Get single course by code."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.course_id, c.course_code,
                   c.course_title, c.venue,
                   c.start_time, c.end_time,
                   c.day, c.lecturer_id,
                   c.created_at, c.updated_at,
                   l.name as lecturer_name
            FROM courses c
            JOIN lecturers l
                ON c.lecturer_id = l.lecturer_id
            WHERE c.course_code = ?
        """, (course_code.upper(),))
        row = cursor.fetchone()
        conn.close()
        return row

    def get_course_by_id(self, course_id):
        """Get course by numeric ID."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM courses WHERE course_id = ?",
            (course_id,))
        row = cursor.fetchone()
        conn.close()
        return row

    # ─────────────────────────────────────────
    # ENROLLMENT METHODS
    # ─────────────────────────────────────────
    def enroll_student(self, student_id,
                        course_code):
        """Enroll student in course via course code."""
        course = self.get_course(course_code)
        if not course:
            return False, "Course not found"

        course_id = course[0]
        conn      = self._get_conn()
        cursor    = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO enrollments
                (student_id, course_id, joined_at)
                VALUES (?, ?, ?)
            """, (student_id, course_id,
                  datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return True, "Enrolled successfully"
        except sqlite3.IntegrityError:
            conn.close()
            return True, "Already enrolled"

    def get_enrolled_courses(self, student_id):
        """Get all courses a student is enrolled in."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.course_id, c.course_code,
                   c.course_title, c.venue,
                   c.start_time, c.end_time,
                   c.day, c.lecturer_id,
                   c.created_at, c.updated_at,
                   l.name as lecturer_name
            FROM courses c
            JOIN enrollments e
                ON c.course_id = e.course_id
            JOIN lecturers l
                ON c.lecturer_id = l.lecturer_id
            WHERE e.student_id = ?
            ORDER BY c.course_code
        """, (student_id,))
        courses = cursor.fetchall()
        conn.close()
        return courses

    def get_enrolled_students(self, course_code):
        """Get all students enrolled in a course."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.* FROM students s
            JOIN enrollments e
                ON s.student_id = e.student_id
            JOIN courses c
                ON e.course_id = c.course_id
            WHERE c.course_code = ?
            ORDER BY s.name
        """, (course_code.upper(),))
        students = cursor.fetchall()
        conn.close()
        return students

    # ─────────────────────────────────────────
    # SYSTEM CONTROL METHODS
    # ─────────────────────────────────────────
    def get_session_status(self, course_code):
        """
        Get current session status for a course.
        Attendance + Feedback auto-open 10 mins
        before end_time, auto-close at end_time.
        Only on the specified day.
        """
        today     = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        day_name  = today.strftime("%A")  # e.g., 'Monday'
        
        # Check if today is the course day
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT day FROM courses
            WHERE course_code = ?
        """, (course_code.upper(),))
        course_row = cursor.fetchone()
        if not course_row or course_row[0] != day_name:
            conn.close()
            return {
                "attendance_open": False,
                "feedback_open"  : False,
                "auto_open_time" : None,
                "auto_close_time": None,
                "overridden"     : False,
                "started_at"     : None,
                "row_id"         : None
            }
        
        cursor.execute("""
            SELECT * FROM system_control
            WHERE course_code = ? AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code.upper(), today_str))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "attendance_open": False,
                "feedback_open"  : False,
                "auto_open_time" : None,
                "auto_close_time": None,
                "overridden"     : False,
                "started_at"     : None,
                "row_id"         : None
            }
        return {
            "attendance_open": bool(row[4]),
            "feedback_open"  : bool(row[5]),
            "auto_open_time" : row[6],
            "auto_close_time": row[7],
            "overridden"     : bool(row[8]),
            "started_at"     : row[9],
            "row_id"         : row[0]
        }

    def upsert_session_control(self, course_code,
                                lecturer_id,
                                attendance_open=None,
                                feedback_open=None,
                                auto_open_time=None,
                                auto_close_time=None,
                                manually_overridden=None,
                                session_started_at=None,
                                session_ended_at=None):
        """Insert or update session control record."""
        today  = datetime.now().strftime("%Y-%m-%d")
        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM system_control
            WHERE course_code = ? AND date = ?
        """, (course_code.upper(), today))
        existing = cursor.fetchone()

        if existing:
            updates = []
            values  = []
            if attendance_open is not None:
                updates.append("attendance_open = ?")
                values.append(int(attendance_open))
            if feedback_open is not None:
                updates.append("feedback_open = ?")
                values.append(int(feedback_open))
            if manually_overridden is not None:
                updates.append("manually_overridden = ?")
                values.append(int(manually_overridden))
            if session_ended_at:
                updates.append("session_ended_at = ?")
                values.append(session_ended_at)
            if updates:
                values.append(existing[0])
                cursor.execute(
                    f"UPDATE system_control SET "
                    f"{', '.join(updates)} WHERE id = ?",
                    values)
        else:
            # Get course_id
            cursor.execute("""
                SELECT course_id FROM courses
                WHERE course_code = ?
            """, (course_code.upper(),))
            cid = cursor.fetchone()
            course_id = cid[0] if cid else None

            cursor.execute("""
                INSERT INTO system_control
                (course_id, course_code, lecturer_id,
                 attendance_open, feedback_open,
                 auto_open_time, auto_close_time,
                 manually_overridden,
                 session_started_at, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course_id,
                course_code.upper(), lecturer_id,
                int(attendance_open or 0),
                int(feedback_open or 0),
                auto_open_time, auto_close_time,
                int(manually_overridden or 0),
                session_started_at or
                datetime.now().isoformat(),
                today
            ))

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # ATTENDANCE METHODS
    # ─────────────────────────────────────────
    def log_attendance(self, student_id,
                        confidence, spoof_verdict,
                        location_verdict,
                        feedback_given,
                        feedback_text,
                        sentiment, sentiment_conf,
                        status, course_code=None,
                        course_title=None):
        """Save attendance to SQLite + Firebase sync."""
        timestamp = datetime.now().isoformat()

        # Get course_id from code
        course_id = None
        if course_code:
            course = self.get_course(course_code)
            if course:
                course_id = course[0]

        conn   = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO attendance
            (student_id, course_id, course_code,
             course_title, confidence, spoof_verdict,
             location_verdict, feedback_given,
             feedback_text, sentiment, sentiment_conf,
             status, synced_firebase, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, 0, ?)
        """, (
            student_id, course_id, course_code,
            course_title, confidence, spoof_verdict,
            location_verdict,
            1 if feedback_given else 0,
            feedback_text if feedback_given
            else "Not Provided",
            sentiment, sentiment_conf,
            status, timestamp
        ))
        record_id = cursor.lastrowid

        # Also save to feedback table if provided
        if feedback_given and feedback_text:
            cursor.execute("""
                INSERT INTO feedback
                (student_id, course_id, course_code,
                 sentiment, text, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                student_id, course_id, course_code,
                sentiment, feedback_text,
                sentiment_conf, timestamp
            ))

        conn.commit()
        conn.close()

        # Firebase sync in background
        if self.firebase_db:
            record = {
                "student_id"      : student_id,
                "hashed_id"       : self.anonymize(
                    student_id),
                "course_code"     : course_code,
                "confidence"      : confidence,
                "status"          : status,
                "sentiment"       : sentiment,
                "timestamp"       : timestamp
            }
            threading.Thread(
                target=self._sync_firebase,
                args=("attendance", record_id, record),
                daemon=True
            ).start()

        return record_id


    def log_attendance_web(self, student_id,
                            course_code,
                            location_verdict,
                            location_method,
                            status,
                            confidence=0.0,
                            spoof_verdict='PENDING'):
        """
        Log attendance from web interface.
        Wrapper around log_attendance for
        GPS/QR-verified attendance records.
        """
        course = self.get_course(course_code)
        course_title = course[2] if course else None

        return self.log_attendance(
            student_id       = student_id,
            confidence       = confidence or 0.0,
            spoof_verdict    = spoof_verdict or 'PENDING',
            location_verdict = (
                str(location_verdict)
                + '(' + str(location_method) + ')'),
            feedback_given   = False,
            feedback_text    = None,
            sentiment        = None,
            sentiment_conf   = None,
            status           = status,
            course_code      = course_code,
            course_title     = course_title
        )

    def save_feedback(self, student_id,
                       course_code,
                       feedback_text,
                       sentiment,
                       sentiment_conf,
                       attendance_record_id=None):
        """
        Save student feedback and sentiment result.
        Updates attendance record if ID provided.
        Also inserts into feedback table.
        """
        import sqlite3 as sql3

        course       = self.get_course(course_code)
        course_id    = course[0] if course else None
        course_title = course[2] if course else None
        timestamp    = datetime.now().isoformat()

        conn   = sql3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Insert into feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id            INTEGER PRIMARY KEY
                              AUTOINCREMENT,
                student_id    TEXT NOT NULL,
                course_id     TEXT,
                course_code   TEXT NOT NULL,
                sentiment     TEXT,
                text          TEXT,
                confidence    REAL,
                timestamp     TEXT NOT NULL
            )
        """)

        cursor.execute("""
            INSERT INTO feedback
            (student_id, course_id, course_code,
             sentiment, text, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (student_id, course_id,
               course_code.upper(),
               sentiment, feedback_text,
               sentiment_conf, timestamp))

        feedback_id = cursor.lastrowid

        # Update attendance record if linked
        if attendance_record_id:
            cursor.execute("""
                UPDATE attendance
                SET feedback_given  = 1,
                    feedback_text   = ?,
                    sentiment       = ?,
                    sentiment_conf  = ?
                WHERE id = ?
                  AND student_id = ?
            """, (feedback_text, sentiment,
                   sentiment_conf,
                   attendance_record_id,
                   student_id))
        else:
            # Find latest attendance record
            # SQLite does not support ORDER BY in UPDATE
            # Find the record ID first, then update it
            cursor.execute("""
            SELECT id FROM attendance
            WHERE student_id   = ?
              AND course_code  = ?
              AND feedback_given = 0
              AND date(timestamp) = date('now')
            ORDER BY timestamp DESC
            LIMIT 1
        """, (student_id, course_code.upper()))
        att_row = cursor.fetchone()
        if att_row:
            cursor.execute("""
                UPDATE attendance
                SET feedback_given = 1,
                    feedback_text  = ?,
                    sentiment      = ?,
                    sentiment_conf = ?
                WHERE id = ?
            """, (feedback_text, sentiment,
                   sentiment_conf, att_row[0]))
        conn.commit()
        conn.close()

        # Firebase sync
        if self.firebase_db:
            record = {
                "student_id"    : student_id,
                "course_code"   : course_code,
                "feedback_text" : feedback_text,
                "sentiment"     : sentiment,
                "confidence"    : sentiment_conf,
                "timestamp"     : timestamp
            }
            import threading
            threading.Thread(
                target=self._sync_firebase,
                args=("feedback", feedback_id,
                       record),
                daemon=True
            ).start()

        return feedback_id

    def save_feedback_skipped(self, student_id,
                               course_code):
        """
        Log that student skipped feedback.
        Sets feedback_given=1, feedback_text=NULL,
        sentiment=NULL.
        """
        import sqlite3 as sql3
        conn   = sql3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Find record first
        cursor.execute("""
            SELECT id FROM attendance
            WHERE student_id   = ?
              AND course_code  = ?
              AND feedback_given = 0
              AND date(timestamp) = date('now')
            ORDER BY timestamp DESC
            LIMIT 1
        """, (student_id, course_code.upper()))
        att_row = cursor.fetchone()
        if att_row:
            cursor.execute("""
                UPDATE attendance
                SET feedback_given = 1,
                    feedback_text  = NULL,
                    sentiment      = 'Skipped',
                    sentiment_conf = NULL
                WHERE id = ?
            """, (att_row[0],))

        updated = cursor.rowcount
        conn.commit()
        conn.close()
        return updated > 0

    def get_student_feedback_history(self,
                                      student_id,
                                      course_code=None):
        """
        Get feedback history for a student.
        Optional filter by course.
        """
        import sqlite3 as sql3
        conn   = sql3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if course_code:
            cursor.execute("""
                SELECT id, course_code,
                       sentiment, text,
                       confidence, timestamp
                FROM feedback
                WHERE student_id  = ?
                  AND course_code = ?
                ORDER BY timestamp DESC
            """, (student_id,
                   course_code.upper()))
        else:
            cursor.execute("""
                SELECT id, course_code,
                       sentiment, text,
                       confidence, timestamp
                FROM feedback
                WHERE student_id = ?
                ORDER BY timestamp DESC
            """, (student_id,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id"         : r[0],
                "course_code": r[1],
                "sentiment"  : r[2],
                "text"       : r[3],
                "confidence" : r[4],
                "timestamp"  : r[5]
            }
            for r in rows
        ]

    def get_course_sentiment_summary(self,
                                      course_code,
                                      date=None):
        """
        Get sentiment breakdown for a course.
        Used for lecturer reports.
        """
        import sqlite3 as sql3
        conn   = sql3.connect(self.DB_PATH)
        cursor = conn.cursor()

        query  = """
            SELECT sentiment,
                   COUNT(*) as cnt,
                   AVG(confidence) as avg_conf,
                   GROUP_CONCAT(text, '|||') as texts
            FROM feedback
            WHERE course_code = ?
              AND sentiment IS NOT NULL
              AND sentiment != 'Skipped'
        """
        params = [course_code.upper()]

        if date:
            query  += " AND date(timestamp) = ?"
            params.append(date)

        query += " GROUP BY sentiment"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        summary = {
            "Positive": {
                "count": 0, "avg_conf": 0,
                "texts": []},
            "Negative": {
                "count": 0, "avg_conf": 0,
                "texts": []},
            "Neutral" : {
                "count": 0, "avg_conf": 0,
                "texts": []}
        }

        for r in rows:
            label = r[0]
            if label in summary:
                texts = []
                if r[3]:
                    texts = [
                        t for t in
                        r[3].split('|||')
                        if t and t.strip()
                    ]
                summary[label] = {
                    "count"   : r[1],
                    "avg_conf": round(r[2] or 0, 3),
                    "texts"   : texts[:5]
                }

        total = sum(
            v["count"] for v in summary.values())
        summary["total"] = total

        return summary

    def get_attendance(self, student_id=None,
                        course_code=None,
                        date=None, limit=100):
        """Get attendance with optional filters."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        query  = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if student_id:
            query += " AND student_id = ?"
            params.append(student_id)
        if course_code:
            query += " AND course_code = ?"
            params.append(course_code.upper())
        if date:
            query += " AND timestamp LIKE ?"
            params.append(f"{date}%")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        records = cursor.fetchall()
        conn.close()
        return records

    # ─────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────
    def get_summary(self):
        """Full database statistics."""
        conn   = self._get_conn()
        cursor = conn.cursor()
        stats  = {}

        tables = [
            "lecturers", "students", "courses",
            "enrollments", "voice_samples",
            "attendance", "feedback",
            "location_log", "system_control",
            "sentiment_feedback"
        ]
        for t in tables:
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {t}")
                stats[t] = cursor.fetchone()[0]
            except Exception:
                stats[t] = 0

        try:
            cursor.execute("""
                SELECT status, COUNT(*)
                FROM attendance GROUP BY status
            """)
            stats["attendance_breakdown"] = dict(
                cursor.fetchall())
            cursor.execute("""
                SELECT sentiment, COUNT(*)
                FROM attendance
                WHERE sentiment IS NOT NULL
                GROUP BY sentiment
            """)
            stats["sentiment_breakdown"] = dict(
                cursor.fetchall())
        except Exception:
            pass

        conn.close()
        return stats

    def display_summary(self):
        """Print database summary."""
        stats = self.get_summary()
        print("\n  📊 Database Summary")
        print("  ─────────────────────────────────────")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k:<28}: {v}")

        mode = ("ENABLED ☁️" if FIREBASE_ENABLED
                else "DISABLED (SQLite only)")
        print(f"\n  Firebase: {mode}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 8: DATABASE MODULE TEST")
    print("=" * 55)

    db = DatabaseModule()
    db.display_summary()

    print("\n  ─────────────────────────────────────")
    print("  🔑 SAMPLE LOGIN CREDENTIALS")
    print("  ─────────────────────────────────────")
    print("  LECTURERS (password: temp123)")
    print("  peter@demo.com  → Dr. Peter (CSC308)")
    print("  anna@demo.com   → Dr. Anna  (CSC306)")
    print("  james@demo.com  → Dr. James (CSC318)")
    print("  grace@demo.com  → Dr. Grace (CSC320)")
    print("  david@demo.com  → Dr. David (CSC322)")
    print("\n  STUDENTS (password: 123456)")
    print("  austin@student.com → Austin (CSC308,306,318)")
    print("  john@student.com   → John   (CSC308,320,322)")
    print("  mary@student.com   → Mary   (CSC306,318,320)")
    print("  paul@student.com   → Paul   (CSC308,306,322)")
    print("  linda@student.com  → Linda  (CSC318,320,322)")
    print("  ─────────────────────────────────────")

    # Test authentication
    print("\n  🔐 Testing Authentication...")
    lecturer = db.login("peter@demo.com", "temp123")
    assert lecturer is not None
    assert lecturer["role"] == "lecturer"
    print(f"  ✅ Lecturer login: {lecturer['name']}")

    student = db.login("austin@student.com", "123456")
    assert student is not None
    assert student["role"] == "student"
    print(f"  ✅ Student login: {student['name']}")

    wrong = db.login("peter@demo.com", "wrongpass")
    assert wrong is None
    print(f"  ✅ Wrong password correctly rejected")

    # Test courses
    courses = db.get_courses(lecturer_id="L001")
    print(f"\n  📚 Dr. Peter's courses: {len(courses)}")
    for c in courses:
        print(f"     {c[1]} — {c[2]}")

    # Test enrollments
    enrolled = db.get_enrolled_courses("STU001")
    print(f"\n  📖 Austin's courses: {len(enrolled)}")
    for c in enrolled:
        print(f"     {c[1]} — {c[2]}")

    print("\n✅ Database Module working!")
