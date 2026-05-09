import os
import sys
import json
import secrets
import sqlite3
import smtplib
import tempfile
import base64
from email.mime.text        import MIMEText
from email.mime.multipart   import MIMEMultipart
from datetime               import datetime, timedelta
from flask                  import (Flask, request,
                                    jsonify,
                                    send_from_directory)
from flask_cors             import CORS

sys.path.append(os.path.abspath("."))
from src.modules.module8_database import DatabaseModule
from src.modules.module11_admin   import AdminModule
from src.modules.system_pipeline  import SystemPipeline

# ─────────────────────────────────────────────
# FFMPEG — AUTO PATH SETUP
# ─────────────────────────────────────────────
_FFMPEG_BIN = (
    r"C:\Users\hp\Downloads"
    r"\ffmpeg-master-latest-win64-gpl-shared"
    r"\ffmpeg-master-latest-win64-gpl-shared"
    r"\bin"
)
_FFMPEG_EXE  = os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
_FFPROBE_EXE = os.path.join(_FFMPEG_BIN, "ffprobe.exe")

# Add to PATH for this process automatically
if os.path.exists(_FFMPEG_EXE):
    os.environ["PATH"] = (
        _FFMPEG_BIN + ";" + os.environ.get("PATH",""))
    try:
        from pydub import AudioSegment
        AudioSegment.converter = _FFMPEG_EXE
        AudioSegment.ffprobe   = _FFPROBE_EXE
        print("✅ FFmpeg auto-configured")
    except ImportError:
        print("⚠️  pydub not installed — "
              "run: pip install pydub")
else:
    print(f"⚠️  FFmpeg not found at: {_FFMPEG_EXE}")
    print("   Update _FFMPEG_BIN in app.py")

# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__,
            static_folder="frontend",
            static_url_path="")
CORS(app)

# Speed up SSL connections
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Log every incoming request using Python's logging module
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.warning("🔍🔍🔍 REQUEST LOGGING ENABLED 🔍🔍🔍")

@app.before_request
def log_request():
    logger.warning(f"➡️ {request.method} {request.path}")
    payload = request.get_json(silent=True)
    if payload:
        logger.warning(f"   Data: {payload}")

@app.after_request
def log_response(response):
    logger.warning(f"⬅️ {request.method} {request.path} → {response.status_code}")
    return response

# ─────────────────────────────────────────────
# EMAIL CONFIGURATION
# ─────────────────────────────────────────────
EMAIL_CONFIG = {
    "ENABLED" : False,
    "HOST"    : "smtp.gmail.com",
    "PORT"    : 587,
    "SENDER"  : "your-email@gmail.com",
    "PASSWORD": "your-app-password",
    "APP_NAME": "AI Voice Attendance System — ABUAD",
    "BASE_URL": "http://localhost:5000"
}

# ─────────────────────────────────────────────
# INITIALIZE SYSTEM
# ─────────────────────────────────────────────
print("=" * 55)
print("   AI VOICE ATTENDANCE SYSTEM")
print("   Starting Backend Server...")
print("=" * 55)

pipeline = SystemPipeline()
db       = DatabaseModule()
admin    = AdminModule()

# ─────────────────────────────────────────────
# DATABASE CONNECTION HELPERS (WAL mode)
# Prevents database locking with concurrent access
# ─────────────────────────────────────────────
def _get_db_conn():
    """Thread-safe SQLite connection for app.py."""
    import sqlite3
    conn = sqlite3.connect(
        db.DB_PATH,
        timeout           = 30,
        check_same_thread = False
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn

# ── Enable WAL mode on startup
# Prevents database locking with multiple threads
def _init_db_wal():
    try:
        import sqlite3 as _sql3
        _conn = _sql3.connect(db.DB_PATH, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA busy_timeout=30000")
        _conn.commit()
        _conn.close()
        print("✅ Database WAL mode enabled")
    except Exception as e:
        print(f"⚠️  WAL mode failed: {e}")

_init_db_wal()

print("\n✅ Backend ready at http://localhost:5000")
print("=" * 55)

# ─────────────────────────────────────────────
# SESSION STORE (in-memory, simple)
# Maps user_id → {email, role, lecturer_id}
# ─────────────────────────────────────────────
active_sessions = {}

# ─────────────────────────────────────────────
# DEMO MODE
# Set True on presentation day to extend
# QR expiry to 5 minutes instead of 30 seconds
# ─────────────────────────────────────────────
DEMO_MODE         = True   # ← set False in production
QR_EXPIRY_SECONDS = 300 if DEMO_MODE else 30
RESET_SECRET      = os.environ.get(
    "SYSTEM_RESET_SECRET",
    "RESET_SYSTEM_AUSTIN258"
)
# ─────────────────────────────────────────────
# BACKGROUND SCHEDULER
# Checks auto-open/close every 60 seconds
# This is what makes the hybrid approach work
# automatically without any user action
# ─────────────────────────────────────────────
from apscheduler.schedulers.background import (
    BackgroundScheduler)

def _auto_check_all_sessions():
    """
    Background job: auto open/close attendance.
    Uses WAL mode connection to avoid locking.
    """
    try:
        conn = _get_db_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT id, course_code,
                   attendance_open,
                   auto_open_time,
                   auto_close_time,
                   manually_overridden
            FROM system_control
            WHERE date = ?
              AND session_ended_at IS NULL
        """, (today,))
        sessions = cursor.fetchall()

        for row in sessions:
            sc_id      = row["id"]
            code       = row["course_code"]
            att_open   = row["attendance_open"]
            auto_open  = row["auto_open_time"]
            auto_close = row["auto_close_time"]
            overridden = row["manually_overridden"]

            if not auto_open or not auto_close:
                continue
            if overridden:
                continue

            try:
                open_dt  = datetime.fromisoformat(
                    auto_open)
                close_dt = datetime.fromisoformat(
                    auto_close)
            except Exception:
                continue

            if now >= open_dt and not att_open:
                cursor.execute("""
                    UPDATE system_control
                    SET attendance_open = 1,
                        feedback_open   = 1
                    WHERE id = ?
                """, (sc_id,))
                conn.commit()
                print(f"  🟢 AUTO-OPENED: {code}")

            elif now >= close_dt and att_open:
                cursor.execute("""
                    UPDATE system_control
                    SET attendance_open  = 0,
                        feedback_open    = 0,
                        session_ended_at = ?
                    WHERE id = ?
                """, (now.isoformat(), sc_id))
                conn.commit()
                print(f"  🔴 AUTO-CLOSED: {code}")

        conn.close()

    except Exception as e:
        print(f"  ⚠️  Scheduler error: {e}")

def _auto_open_by_course_time():
    """
    Auto-open attendance based on course schedule.
    Handles cases where lecturer did not click
    Start Session.
    """
    try:
        conn = _get_db_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        now   = datetime.now()
        today = now.strftime("%Y-%m-%d")

        cursor.execute("""
            SELECT course_code, start_time,
                   end_time, lecturer_id
            FROM courses
        """)
        courses = cursor.fetchall()

        for row in courses:
            code        = row["course_code"]
            end_str     = row["end_time"]
            lecturer_id = row["lecturer_id"]

            if not end_str:
                continue

            try:
                end_dt  = datetime.strptime(
                    f"{today} {end_str}",
                    "%Y-%m-%d %H:%M")
                open_dt = end_dt - timedelta(
                    minutes=10)
            except Exception:
                continue

            in_window = open_dt <= now <= end_dt

            cursor.execute("""
                SELECT id, attendance_open,
                       manually_overridden,
                       session_ended_at
                FROM system_control
                WHERE course_code = ?
                  AND date = ?
                ORDER BY id DESC LIMIT 1
            """, (code, today))
            sc_row = cursor.fetchone()

            if in_window:
                if not sc_row:
                    cursor.execute("""
                        INSERT INTO system_control
                        (course_id, course_code,
                         lecturer_id,
                         attendance_open,
                         feedback_open,
                         auto_open_time,
                         auto_close_time,
                         manually_overridden,
                         session_started_at, date)
                        VALUES
                        (NULL,?,?,1,1,?,?,0,?,?)
                    """, (code, lecturer_id,
                           open_dt.isoformat(),
                           end_dt.isoformat(),
                           now.isoformat(),
                           today))
                    conn.commit()
                    print(f"  🟢 AUTO-SESSION: {code}")

                elif (not sc_row["attendance_open"]
                      and not sc_row[
                          "manually_overridden"]
                      and not sc_row[
                          "session_ended_at"]):
                    cursor.execute("""
                        UPDATE system_control
                        SET attendance_open = 1,
                            feedback_open   = 1
                        WHERE id = ?
                    """, (sc_row["id"],))
                    conn.commit()
                    print(f"  🟢 AUTO-OPENED: {code}")

            elif (sc_row and
                  sc_row["attendance_open"] and
                  not sc_row["manually_overridden"]
                  and now > end_dt):
                cursor.execute("""
                    UPDATE system_control
                    SET attendance_open  = 0,
                        feedback_open    = 0,
                        session_ended_at = ?
                    WHERE id = ?
                """, (now.isoformat(),
                       sc_row["id"]))
                conn.commit()
                print(f"  🔴 AUTO-CLOSED: {code}")

        conn.close()

    except Exception as e:
        print(f"  ⚠️  Auto-time error: {e}")

def _get_session(user_id):
    """Get active session for a user."""
    return active_sessions.get(str(user_id))

def _set_session(user_id, user_data):
    """Store session after login."""
    active_sessions[str(user_id)] = user_data

def _clear_session(user_id):
    """Remove session on logout."""
    active_sessions.pop(str(user_id), None)

def _require_lecturer(user_id, course_code):
    """
    Verify user is a logged-in lecturer
    who owns the given course.
    Returns (ok, lecturer_id or error_response).
    """
    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return False, jsonify({
            "success": False,
            "message": "Authentication required"
        }), 401

    course = db.get_course(course_code)
    if not course:
        return False, jsonify({
            "success": False,
            "message": f"Course {course_code} not found"
        }), 404

    if course[7] != session["lecturer_id"]:
        return False, jsonify({
            "success": False,
            "message": "Access denied — "
                       "you do not own this course"
        }), 403

    return True, session["lecturer_id"]

# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend",
                                "index.html")

@app.route("/<path:path>")
def serve_static(path):
    full = os.path.join("frontend", path)
    if os.path.exists(full):
        return send_from_directory("frontend", path)
    return send_from_directory("frontend",
                                "index.html")

# ─────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    logger.warning("🔥 LOGIN ROUTE CALLED!")
    data     = request.get_json()
    email    = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password required"
        }), 400

    user = db.login(email, password)
    if not user:
        return jsonify({
            "success": False,
            "message": "Invalid email or password"
        }), 401

    # Log login time
    db.log_login(user["id"], user["role"])

    # Store session
    _set_session(user["id"], {
        "email"      : user["email"],
        "role"       : user["role"],
        "lecturer_id": user["id"]
                       if user["role"] == "lecturer"
                       else None
    })

    # ── ADD enrollment check before return ──
    enrolled = db.get_enrollment_status(user["id"])

    return jsonify({
        "success": True,
        "message": f"Welcome, {user['name']}!",
        "user"   : {
            "id"            : user["id"],
            "name"          : user["name"],
            "email"         : user["email"],
            "role"          : user["role"],
            "first_login"   : user.get(
                "first_login", False),
            "matric_number" : user.get(
                "matric_number"),
            "voice_enrolled": enrolled["enrolled"]
        }
    })

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    data    = request.get_json() or {}
    user_id = data.get("user_id")
    if user_id:
        _clear_session(user_id)
    return jsonify({
        "success": True,
        "message": "Logged out"
    })

# ─────────────────────────────────────────────
# STUDENT REGISTRATION
# ─────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register_student():
    """
    Student self-registration.

    Validates:
    - Full name (min 2 chars)
    - Email format + uniqueness
    - Matric number format: YY/SCI01/NNN
    - Password strength (strong required)
    - Confirm password match

    On success:
    - Creates student account
    - voice_template = NULL (enrolled later)
    - Returns student info for auto-login
    """
    data            = request.get_json()
    name            = data.get("name", "").strip()
    email           = data.get("email", "").strip()
    matric_number   = data.get(
        "matric_number", "").strip()
    password        = data.get("password", "").strip()
    confirm         = data.get(
        "confirm_password", "").strip()

    # ── Basic presence check
    if not all([name, email,
                matric_number, password, confirm]):
        return jsonify({
            "success": False,
            "message": "All fields are required"
        }), 400

    # ── Password match
    if password != confirm:
        return jsonify({
            "success": False,
            "message": "Passwords do not match"
        }), 400

    # ── Register (all other validation inside)
    success, message, student_id = (
        db.register_student(
            name, email, matric_number, password))

    if not success:
        return jsonify({
            "success": False,
            "message": message
        }), 400

    # ── Auto-login after registration
    user = db.login(email, password)
    if user:
        db.log_login(user["id"], user["role"])
        _set_session(user["id"], {
            "email"      : user["email"],
            "role"       : user["role"],
            "lecturer_id": None
        })

    return jsonify({
        "success"       : True,
        "message"       : message,
        "student_id"    : student_id,
        "voice_enrolled": False,
        "user"          : {
            "id"           : student_id,
            "name"         : name,
            "email"        : email,
            "role"         : "student",
            "matric_number": matric_number.upper(),
            "voice_enrolled": False
        }
    }), 201

# ─────────────────────────────────────────────
# FORGOT PASSWORD
# ─────────────────────────────────────────────
def send_reset_email(to_email, token, role):
    reset_link = (f"{EMAIL_CONFIG['BASE_URL']}"
                  f"/reset-password?token={token}")
    print(f"\n  📧 PASSWORD RESET")
    print(f"  To    : {to_email}")
    print(f"  Role  : {role}")
    print(f"  Link  : {reset_link}")
    print(f"  Token : {token}")

    if not EMAIL_CONFIG["ENABLED"]:
        print("  ℹ️  Email disabled — use token above")
        return True

    try:
        body = (
            f"Hello,\n\nReset your password:\n\n"
            f"{reset_link}\n\n"
            f"Expires in 1 hour.\n\n"
            f"— {EMAIL_CONFIG['APP_NAME']}"
        )
        msg            = MIMEMultipart()
        msg["From"]    = EMAIL_CONFIG["SENDER"]
        msg["To"]      = to_email
        msg["Subject"] = f"Password Reset — " \
                         f"{EMAIL_CONFIG['APP_NAME']}"
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EMAIL_CONFIG["HOST"],
                           EMAIL_CONFIG["PORT"]) as s:
            s.starttls()
            s.login(EMAIL_CONFIG["SENDER"],
                    EMAIL_CONFIG["PASSWORD"])
            s.sendmail(EMAIL_CONFIG["SENDER"],
                       to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"  ❌ Email error: {e}")
        return False

@app.route("/api/auth/forgot-password",
           methods=["POST"])
def forgot_password():
    data  = request.get_json()
    email = data.get("email", "").strip().lower()

    if not email:
        return jsonify({
            "success": False,
            "message": "Email is required"
        }), 400

    exists, role = db.email_exists(email)
    if not exists:
        return jsonify({
            "success": True,
            "message": ("If that email is registered, "
                        "you'll receive a reset link")
        })

    token      = secrets.token_urlsafe(32)
    expires_at = (datetime.now() +
                  timedelta(hours=1)).isoformat()
    db.store_reset_token(email, token, expires_at)
    send_reset_email(email, token, role)
    db.cleanup_expired_tokens()

    return jsonify({
        "success"  : True,
        "message"  : ("If that email is registered, "
                      "you'll receive a reset link"),
        "dev_token": token,
        "dev_link" : (f"{EMAIL_CONFIG['BASE_URL']}"
                      f"/reset-password?token={token}")
    })

@app.route("/api/auth/verify-token",
           methods=["POST"])
def verify_token():
    data  = request.get_json()
    token = data.get("token", "").strip()

    if not token:
        return jsonify({
            "success": False,
            "message": "Token required"
        }), 400

    token_data = db.verify_reset_token(token)
    if not token_data:
        return jsonify({
            "success": False,
            "message": "Invalid or expired link"
        }), 400

    return jsonify({
        "success": True,
        "email"  : token_data["email"],
        "role"   : token_data["role"]
    })

@app.route("/api/auth/reset-password",
           methods=["POST"])
def reset_password():
    data     = request.get_json()
    token    = data.get("token", "").strip()
    new_pw   = data.get("new_password", "").strip()
    confirm  = data.get("confirm_password", "").strip()

    if not token or not new_pw:
        return jsonify({
            "success": False,
            "message": "Token and new password required"
        }), 400

    if new_pw != confirm:
        return jsonify({
            "success": False,
            "message": "Passwords do not match"
        }), 400

    if len(new_pw) < 6:
        return jsonify({
            "success": False,
            "message": "Minimum 6 characters"
        }), 400

    success, message = db.reset_password(token, new_pw)
    return jsonify({"success": success,
                    "message": message})

@app.route("/api/auth/change-password",
           methods=["POST"])
def change_password():
    """
    Change password.
    Used for:
    1. First-login forced password change (lecturer)
    2. Profile password update (all users)

    For first-login:
    - Old password = temporary password
    - New password must be strong
    - first_login flag set to 0 after change
    - Returns redirect_to_login = True
      so frontend logs them out and
      redirects to login page
    """
    data         = request.get_json()
    email        = data.get("email", "").strip()
    old_pw       = data.get("old_password", "").strip()
    new_pw       = data.get("new_password", "").strip()
    confirm      = data.get(
        "confirm_password", "").strip()
    is_first     = data.get(
        "is_first_login", False)

    if not email or not old_pw or not new_pw:
        return jsonify({
            "success": False,
            "message": "All fields are required"
        }), 400

    # Verify old password
    user = db.login(email, old_pw)
    if not user:
        return jsonify({
            "success": False,
            "message": "Current password is incorrect"
        }), 401

    # Confirm match
    if new_pw != confirm:
        return jsonify({
            "success": False,
            "message": "New passwords do not match"
        }), 400

    # Reject same password
    if old_pw == new_pw:
        return jsonify({
            "success": False,
            "message": "New password must be different "
                       "from your current password"
        }), 400

    # Enforce strength for first login
    is_strong, strength_msg, score = (
        db.check_password_strength(new_pw))

    if not is_strong:
        return jsonify({
            "success"       : False,
            "message"       : strength_msg,
            "strength_score": score
        }), 400

    # Update password
    success = db.update_password_direct(
        email,
        new_pw,
        mark_first_login_done=True
    )

    if success:
        # Clear session so they must re-login
        _clear_session(user["id"])

    return jsonify({
        "success"          : success,
        "message"          : (
            "Password updated successfully! "
            "Please log in with your new password."
            if success else "Update failed"),
        "redirect_to_login": True,
        "is_first_login"   : is_first
    })

# ─────────────────────────────────────────────
# PROFILE ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/profile/get", methods=["GET"])
def get_profile():
    user_id = request.args.get("user_id")
    role    = request.args.get("role")

    if not user_id or not role:
        return jsonify({
            "success": False,
            "message": "user_id and role required"
        }), 400

    profile = db.get_user_profile(user_id, role)
    if not profile:
        return jsonify({
            "success": False,
            "message": "User not found"
        }), 404

    profile["courses"] = db.get_user_course_list(
        user_id, role)
    profile["course_count"] = len(profile["courses"])

    if role == "student":
        profile["attendance_stats"] = (
            db.get_user_attendance_stats(user_id))
        att_counts = (
            db.get_student_course_attendance_counts(
                user_id))
        for c in profile["courses"]:
            c["attendance_count"] = att_counts.get(
                c["course_code"], 0)

    return jsonify({"success": True,
                    "profile": profile})

@app.route("/api/profile/update-name",
           methods=["PUT"])
def update_profile_name():
    data     = request.get_json()
    user_id  = data.get("user_id")
    role     = data.get("role")
    new_name = data.get("name", "").strip()

    if not all([user_id, role, new_name]):
        return jsonify({
            "success": False,
            "message": "user_id, role and name required"
        }), 400

    success, message = db.update_profile_name(
        user_id, role, new_name)
    return jsonify({"success": success,
                    "message": message})

@app.route("/api/profile/update-email",
           methods=["PUT"])
def update_profile_email():
    """
    Update email for student or lecturer.
    Validates format and uniqueness.
    """
    data      = request.get_json()
    user_id   = data.get("user_id")
    role      = data.get("role")
    new_email = data.get("email", "").strip().lower()

    if not all([user_id, role, new_email]):
        return jsonify({
            "success": False,
            "message": "user_id, role and email required"
        }), 400

    success, message = db.update_email(
        user_id, role, new_email)

    if success:
        # Update session email
        session = _get_session(user_id)
        if session:
            session["email"] = new_email
            _set_session(user_id, session)

    return jsonify({"success": success,
                    "message": message})

@app.route("/api/profile/upload-picture",
           methods=["POST"])
def upload_profile_picture():
    user_id = request.form.get("user_id")
    pic     = request.files.get("picture")

    if not user_id or not pic:
        return jsonify({
            "success": False,
            "message": "user_id and picture required"
        }), 400

    pic.seek(0, 2)
    size = pic.tell()
    pic.seek(0)

    if size > 2 * 1024 * 1024:
        return jsonify({
            "success": False,
            "message": "Image too large — max 2MB"
        }), 400

    allowed = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
    ext = (pic.filename.rsplit('.', 1)[-1].lower()
           if '.' in pic.filename else 'jpg')

    if ext not in allowed:
        return jsonify({
            "success": False,
            "message": "Only JPG, PNG, GIF, WEBP allowed"
        }), 400

    pic_data = base64.b64encode(
        pic.read()).decode('utf-8')
    db.save_profile_picture(user_id, pic_data, ext)

    return jsonify({
        "success" : True,
        "message" : "Profile picture updated!",
        "pic_data": (f"data:image/{ext};"
                     f"base64,{pic_data}")
    })

@app.route("/api/profile/remove-picture",
           methods=["DELETE"])
def remove_profile_picture():
    data    = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({
            "success": False,
            "message": "user_id required"
        }), 400

    success, message = db.remove_profile_picture(
        user_id)
    return jsonify({"success": success,
                    "message": message})

@app.route("/api/profile/update-matric",
           methods=["PUT"])
def update_matric():
    data       = request.get_json()
    student_id = data.get("user_id")
    matric     = data.get("matric_number", "").strip()
    role       = data.get("role", "")

    if role != "student":
        return jsonify({
            "success": False,
            "message": "Students only"
        }), 403

    if not student_id or not matric:
        return jsonify({
            "success": False,
            "message": "student_id and matric required"
        }), 400

    success, message = db.update_matric_number(
        student_id, matric)
    return jsonify({"success": success,
                    "message": message})

# ─────────────────────────────────────────────
# STUDENT ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/student/courses", methods=["GET"])
def get_student_courses():
    student_id = request.args.get("student_id")
    if not student_id:
        return jsonify({
            "success": False,
            "message": "student_id required"
        }), 400

    courses    = db.get_enrolled_courses(student_id)
    att_counts = (
        db.get_student_course_attendance_counts(
            student_id))
    result     = []

    for c in courses:
        status = db.get_session_status(c[1])
        result.append({
            "course_id"       : c[0],
            "course_code"     : c[1],
            "course_title"    : c[2],
            "venue"           : c[3],
            "start_time"      : c[4],
            "end_time"        : c[5],
            "day"             : c[6],
            "lecturer_id"     : c[7],
            "lecturer_name"   : (c[-1]
                                 if len(c) > 9
                                 else "N/A"),
            "attendance_open" : status[
                "attendance_open"],
            "feedback_open"   : status["feedback_open"],
            "auto_open_time"  : status.get(
                "auto_open_time"),
            "auto_close_time" : status.get(
                "auto_close_time"),
            "attendance_count": att_counts.get(
                c[1], 0)
        })

    return jsonify({"success": True,
                    "courses": result})

@app.route("/api/student/enroll", methods=["POST"])
def enroll_student():
    data        = request.get_json()
    student_id  = data.get("student_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and course_code required"
        }), 400

    # Verify course exists
    course = db.get_course(course_code)
    if not course:
        return jsonify({
            "success": False,
            "message": f"Course '{course_code}' "
                       f"not found"
        }), 404

    ok, msg = db.enroll_student(student_id, course_code)
    return jsonify({"success": ok, "message": msg})

@app.route("/api/student/unenroll",
           methods=["DELETE"])
def unenroll_student():
    """Remove student from a course."""
    data        = request.get_json()
    student_id  = data.get("student_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and course_code required"
        }), 400

    success, message = db.remove_course_enrollment(
        student_id, course_code)
    return jsonify({"success": success,
                    "message": message})

# ─────────────────────────────────────────────
# ATTENDANCE ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/attendance/status", methods=["GET"])
def attendance_status():
    course_code = request.args.get(
        "course_code", "").upper()
    if not course_code:
        return jsonify({
            "success": False,
            "message": "course_code required"
        }), 400

    status = db.get_session_status(course_code)
    return jsonify({
        "success"         : True,
        "attendance_open" : status["attendance_open"],
        "feedback_open"   : status["feedback_open"],
        "auto_open_time"  : status.get("auto_open_time"),
        "auto_close_time" : status.get("auto_close_time")
    })

@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    """
    Voice verification step.
    Called after location is already verified
    (GPS or QR). Only does:
    1. Audio preprocessing
    2. Anti-spoofing
    3. Voice recognition (CNN-LSTM)
    4. Updates the existing attendance record

    Does NOT re-run location check.
    Location was already verified in mark-web.
    """
    student_id  = request.form.get(
        "student_id", "unknown")
    course_code = request.form.get(
        "course_code", "").upper()
    record_id   = request.form.get("record_id")
    audio_file  = request.files.get("audio")

    if not audio_file or not course_code:
        return jsonify({
            "success": False,
            "message": "audio and course_code required"
        }), 400

    # Save temp WebM file
    tmp_webm = tempfile.NamedTemporaryFile(
        suffix=".webm", delete=False)
    audio_file.save(tmp_webm.name)
    tmp_webm.close()

    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False)
    tmp_wav.close()

    try:
        # ── Convert WebM → WAV
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(
                tmp_webm.name)
            seg = seg.set_frame_rate(16000)
            seg = seg.set_channels(1)
            seg.export(tmp_wav.name, format="wav")
            wav_path = tmp_wav.name
            print(f"  ✅ Audio converted: "
                  f"{len(seg)/1000:.1f}s")
        except Exception as e:
            print(f"  ⚠️  Conversion failed: {e}")
            wav_path = tmp_webm.name

        # ── Preprocess audio
        from src.preprocessing.preprocess import (
            preprocess_audio, extract_mfcc)
        import numpy as np

        audio, sr = preprocess_audio(wav_path)

        if audio is None or len(audio) == 0:
            return jsonify({
                "success": False,
                "status" : "REJECTED",
                "reason" : "Could not process audio. "
                           "Please speak clearly.",
                "confidence": 0.0
            }), 400

        # Quality check
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            return jsonify({
                "success": False,
                "status" : "REJECTED",
                "reason" : "Audio too quiet. "
                           "Speak louder and closer "
                           "to the microphone.",
                "confidence": 0.0
            }), 400

        # ── Anti-Spoofing
        spoof_verdict  = "LIVE"
        spoof_detected = False
        try:
            from src.preprocessing.anti_spoofing \
                import AntiSpoofing
            spoof_checker = AntiSpoofing()
            spoof_result  = spoof_checker.check(audio,
                                                 sr)
            if spoof_result.get("verdict") == "SPOOFED":
                spoof_detected = True
                spoof_verdict  = "SPOOFED"
                print(f"  ⚠️  Spoof detected for "
                      f"{student_id}")
        except Exception as e:
            print(f"  ⚠️  Anti-spoof skipped: {e}")

        if spoof_detected:
            # Update record as rejected
            if record_id:
                conn   = _get_db_conn()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE attendance
                    SET status        = 'REJECTED',
                        spoof_verdict = 'SPOOFED',
                        confidence    = 0.0
                    WHERE id         = ?
                      AND student_id  = ?
                """, (record_id, student_id))
                conn.commit()
                conn.close()

            return jsonify({
                "success"   : False,
                "status"    : "REJECTED",
                "reason"    : "Spoofed audio detected. "
                              "Use your real voice.",
                "confidence": 0.0,
                "spoof"     : "SPOOFED"
            })

        # ── Voice Verification (CNN-LSTM)
        confidence  = 0.0
        verified    = False
        reason      = "Voice not recognized"

        try:
            # Extract MFCC from live audio
            live_mfcc = extract_mfcc(audio, sr)

            # Load enrolled template
            template_data = db.get_voice_template(
                student_id)

            if not template_data:
                return jsonify({
                    "success"   : False,
                    "status"    : "REJECTED",
                    "reason"    : "Voice not enrolled. "
                                  "Please enroll first.",
                    "confidence": 0.0
                })

            enrolled_features = template_data[
                "features"]

            # Compare live MFCC vs enrolled features
            # Simple cosine similarity across samples
            similarities = []
            for enrolled_mfcc in enrolled_features:
                try:
                    enrolled_arr = np.array(
                        enrolled_mfcc)
                    live_arr     = np.array(live_mfcc)

                    # Align lengths
                    min_len = min(
                        enrolled_arr.shape[-1],
                        live_arr.shape[-1])
                    e = enrolled_arr[..., :min_len]
                    l = live_arr[..., :min_len]

                    # Flatten and compute similarity
                    e_flat = e.flatten()
                    l_flat = l.flatten()

                    # Cosine similarity
                    dot    = float(np.dot(
                        e_flat, l_flat))
                    norm_e = float(np.linalg.norm(
                        e_flat))
                    norm_l = float(np.linalg.norm(
                        l_flat))

                    if norm_e > 0 and norm_l > 0:
                        sim = dot / (norm_e * norm_l)
                        similarities.append(
                            max(0.0, sim))
                except Exception as se:
                    print(f"  ⚠️  Sample compare "
                          f"failed: {se}")
                    continue

            if not similarities:
                confidence = 0.0
                verified   = False
                reason     = "Could not compare voice"
            else:
                # Average similarity → confidence %
                avg_sim    = float(np.mean(
                    similarities))
                confidence = round(
                    min(avg_sim * 100, 99.9), 2)
                # Threshold: 45% confidence
                THRESHOLD  = 45.0
                verified   = confidence >= THRESHOLD
                reason     = (
                    f"Voice matched at {confidence:.1f}%"
                    if verified else
                    f"Voice mismatch — "
                    f"confidence {confidence:.1f}% "
                    f"(need ≥{THRESHOLD}%)")

            print(f"  {'✅' if verified else '❌'} "
                  f"Voice: {confidence:.1f}% "
                  f"from {len(similarities)} samples")

        except Exception as e:
            print(f"  ⚠️  Voice verify error: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: try full pipeline
            try:
                result = pipeline.run_attendance(
                    student_lat     = 0,
                    student_lon     = 0,
                    student_id      = student_id,
                    course_code     = course_code,
                    test_audio_path = wav_path,
                    test_course     = course_code,
                    skip_location   = True
                )
                confidence = float(result.get(
                    "confidence", 0))
                verified   = result.get(
                    "status") == "PRESENT"
                reason     = result.get(
                    "reason", "")
            except Exception as e2:
                print(f"  ❌ Pipeline fallback "
                      f"failed: {e2}")

        # ── Update attendance record
        final_status = "PRESENT" if verified \
                       else "REJECTED"

        if record_id:
            conn   = _get_db_conn()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE attendance
                SET status        = ?,
                    confidence    = ?,
                    spoof_verdict = ?
                WHERE id         = ?
                  AND student_id  = ?
            """, (final_status,
                   confidence,
                   spoof_verdict,
                   record_id,
                   student_id))
            conn.commit()
            conn.close()
            print(f"  {'✅' if verified else '❌'} "
                  f"Attendance {final_status}: "
                  f"{student_id} "
                  f"({confidence:.1f}%)")
        else:
            # No record_id — create fresh record
            db.log_attendance_web(
                student_id       = student_id,
                course_code      = course_code,
                location_verdict = "VOICE_ONLY",
                location_method  = "VOICE",
                status           = final_status,
                confidence       = confidence,
                spoof_verdict    = spoof_verdict
            )

        return jsonify({
            "success"   : verified,
            "status"    : final_status,
            "student_id": student_id,
            "confidence": confidence,
            "spoof"     : spoof_verdict,
            "course"    : course_code,
            "reason"    : reason if not verified
                          else "Voice verified ✅",
            "timestamp" : datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "status" : "REJECTED",
            "reason" : f"Processing error: {str(e)}",
            "confidence": 0.0
        }), 500

    finally:
        for tmp in [tmp_webm.name, tmp_wav.name]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

def attendance_history():
    student_id  = request.args.get("student_id")
    course_code = request.args.get("course_code")
    date        = request.args.get("date")

    records = db.get_attendance(
        student_id  = student_id,
        course_code = course_code,
        date        = date
    )

    result = []
    for r in records:
        result.append({
            "id"              : r[0],
            "student_id"      : r[1],
            "course_code"     : r[3],
            "course_title"    : r[4],
            "confidence"      : r[5],
            "spoof_verdict"   : r[6],
            "location_verdict": r[7],
            "feedback_given"  : bool(r[8]),
            "feedback_text"   : r[9],
            "sentiment"       : r[10],
            "sentiment_conf"  : r[11],
            "status"          : r[12],
            "timestamp"       : r[14]
        })

    return jsonify({"success": True,
                    "records": result,
                    "total"  : len(result)})

# ─────────────────────────────────────────────
# FEEDBACK ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/feedback/status", methods=["GET"])
def feedback_status():
    course_code = request.args.get(
        "course_code", "").upper()
    status = db.get_session_status(course_code)
    return jsonify({
        "success"      : True,
        "feedback_open": status["feedback_open"]
    })

@app.route("/api/feedback/submit",
           methods=["POST"])
def submit_feedback():
    """
    Process voice feedback submission.
    Bypasses pipeline._capture_audio entirely.
    Handles WebM → WAV → STT → Sentiment directly.
    """
    student_id  = request.form.get("student_id")
    course_code = request.form.get(
        "course_code", "").upper()
    record_id   = request.form.get("record_id")
    audio_file  = request.files.get("audio")

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and "
                       "course_code required"
        }), 400

    # Check feedback window
    status = db.get_session_status(course_code)
    if not status.get("feedback_open"):
        return jsonify({
            "success": False,
            "message": "Feedback window is currently "
                       "CLOSED for this course."
        }), 403

    if not audio_file:
        return jsonify({
            "success": False,
            "message": "No audio file received"
        }), 400

    # Save temp files
    tmp_webm = tempfile.NamedTemporaryFile(
        suffix=".webm", delete=False)
    audio_file.save(tmp_webm.name)
    tmp_webm.close()

    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False)
    tmp_wav.close()

    try:
        # ── Step 1: Convert WebM → WAV
        wav_path = tmp_webm.name  # fallback
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(
                tmp_webm.name)
            seg = seg.set_frame_rate(16000)
            seg = seg.set_channels(1)
            seg.export(tmp_wav.name,
                       format="wav")
            wav_path = tmp_wav.name
            print(f"  ✅ Feedback audio: "
                  f"{len(seg)/1000:.1f}s")
        except Exception as e:
            print(f"  ⚠️  Conversion failed: {e}")
            wav_path = tmp_webm.name

        # ── Step 2: Preprocess audio
        try:
            from src.preprocessing.preprocess \
                import preprocess_audio
            audio, sr = preprocess_audio(wav_path)
        except Exception as e:
            print(f"  ⚠️  Preprocess failed: {e}")
            audio, sr = None, 16000

        if audio is None or len(audio) == 0:
            return jsonify({
                "success": False,
                "message": "Could not process audio. "
                           "Please speak clearly."
            }), 400

        # ── Step 3: Speech-to-Text
        feedback_text = None

        # Try Google STT
        try:
            import speech_recognition as sr_lib
            recognizer = sr_lib.Recognizer()
            with sr_lib.AudioFile(wav_path) as src:
                recognizer.adjust_for_ambient_noise(
                    src, duration=0.3)
                audio_data = recognizer.record(src)
            feedback_text = recognizer.recognize_google(
                audio_data)
            print(f"  ✅ STT: {feedback_text[:60]}")
        except Exception as e:
            print(f"  ⚠️  Google STT failed: {e}")

        # Try Sphinx offline fallback
        if not feedback_text:
            try:
                import speech_recognition as sr_lib
                recognizer = sr_lib.Recognizer()
                with sr_lib.AudioFile(wav_path) as src:
                    audio_data = recognizer.record(src)
                feedback_text = \
                    recognizer.recognize_sphinx(
                        audio_data)
                print(f"  ✅ Sphinx: {feedback_text}")
            except Exception as e:
                print(f"  ⚠️  Sphinx failed: {e}")

        # Use placeholder if both STT fail
        if not feedback_text or \
                len(feedback_text.strip()) < 2:
            feedback_text = "[Voice feedback recorded]"
            print("  ℹ️  Using placeholder text")

        # ── Step 4: Sentiment Analysis
        sentiment   = "Neutral"
        confidence  = 0.60

        # Try trained model first
        # Try trained model first
        try:
            sent_result = pipeline._classify_sentiment(
                feedback_text)
            if sent_result:
                sentiment  = sent_result.get(
                    "sentiment", "Neutral")
                raw_conf   = float(sent_result.get(
                    "confidence", 0.60))

                # Normalize: model may return
                # 0.56 (fraction) or 56.46 (percent)
                # We always store as 0.0–1.0
                if raw_conf > 1.0:
                    confidence = raw_conf / 100.0
                else:
                    confidence = raw_conf

                # Clamp to valid range
                confidence = max(0.0,
                                 min(1.0, confidence))

                print(f"  ✅ Model sentiment: "
                      f"{sentiment} "
                      f"({confidence:.2%})")
        except Exception as e:
            print(f"  ⚠️  Model sentiment "
                  f"failed: {e}")

            # Keyword fallback
            text_lower = feedback_text.lower()
            pos_words  = [
                "good", "great", "excellent",
                "amazing", "wonderful", "helpful",
                "clear", "best", "love", "enjoy",
                "understand", "perfect", "thank",
                "well", "nice", "fantastic",
                "interesting", "brilliant"
            ]
            neg_words  = [
                "bad", "poor", "terrible", "boring",
                "confusing", "hard", "difficult",
                "unclear", "slow", "worst", "hate",
                "awful", "frustrating", "disappointed",
                "waste", "improve", "problem"
            ]
            pos = sum(1 for w in pos_words
                      if w in text_lower)
            neg = sum(1 for w in neg_words
                      if w in text_lower)

            if pos > neg:
                sentiment  = "Positive"
                confidence = min(
                    0.5 + pos/(pos+neg+1) * 0.45,
                    0.95)
            elif neg > pos:
                sentiment  = "Negative"
                confidence = min(
                    0.5 + neg/(pos+neg+1) * 0.45,
                    0.95)
            else:
                sentiment  = "Neutral"
                confidence = 0.60

            print(f"  ✅ Keyword sentiment: "
                  f"{sentiment} ({confidence:.2%})")

        # ── Step 5: Save to database
        feedback_id = db.save_feedback(
            student_id          = student_id,
            course_code         = course_code,
            feedback_text       = feedback_text,
            sentiment           = sentiment,
            sentiment_conf      = confidence,
            attendance_record_id= (
                int(record_id)
                if record_id else None)
        )

        icons = {
            "Positive": "😊",
            "Negative": "😟",
            "Neutral" : "😐"
        }

        print(f"  ✅ Feedback saved: "
              f"{sentiment} for {student_id}")

        return jsonify({
            "success"       : True,
            "message"       : "Feedback submitted!",
            "feedback_id"   : feedback_id,
            "feedback_text" : feedback_text,
            "sentiment"     : sentiment,
            "confidence"    : confidence,
            "sentiment_icon": icons.get(
                sentiment, "😐")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Feedback processing "
                       f"failed: {str(e)}"
        }), 500

    finally:
        for tmp in [tmp_webm.name, tmp_wav.name]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

@app.route("/api/feedback/skip", methods=["POST"])
def skip_feedback():
    data = request.get_json() or {}
    student_id = data.get("student_id")
    course_code = data.get("course_code", "").upper()

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and course_code are required"
        }), 400

    updated = db.save_feedback_skipped(
        student_id=student_id,
        course_code=course_code
    )

    if updated is False:
        return jsonify({
            "success": False,
            "message": "No matching attendance record found to skip feedback."
        }), 404

    return jsonify({
        "success": True,
        "message": "Feedback skipped successfully.",
        "status": "SKIPPED"
    })

# ─────────────────────────────────────────────
# COURSES (ALL)
# ─────────────────────────────────────────────
@app.route("/api/courses/all", methods=["GET"])
def get_all_courses():
    """Get all available courses for enrollment."""
    courses = db.get_courses()
    result  = []
    for c in courses:
        status = db.get_session_status(c[1])
        result.append({
            "course_id"       : c[0],
            "course_code"     : c[1],
            "course_title"    : c[2],
            "venue"           : c[3],
            "start_time"      : c[4],
            "end_time"        : c[5],
            "lecturer_name"   : (c[-1]
                                 if len(c) > 9
                                 else "N/A"),
            "attendance_open" : status[
                "attendance_open"],
            "feedback_open"   : status["feedback_open"]
        })
    return jsonify({"success": True,
                    "courses": result})

# ─────────────────────────────────────────────
# LECTURER ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/lecturer/courses", methods=["GET"])
def get_lecturer_courses():
    lecturer_id = request.args.get("lecturer_id")
    if not lecturer_id:
        return jsonify({
            "success": False,
            "message": "lecturer_id required"
        }), 400

    courses = db.get_courses(lecturer_id=lecturer_id)
    result  = []
    for c in courses:
        status = db.get_session_status(c[1])
        result.append({
            "course_id"       : c[0],
            "course_code"     : c[1],
            "course_title"    : c[2],
            "venue"           : c[3],
            "start_time"      : c[4],
            "end_time"        : c[5],
            "day"             : c[6],
            "lecturer_id"     : c[7],
            "attendance_open" : status[
                "attendance_open"],
            "feedback_open"   : status["feedback_open"],
            "auto_open_time"  : status.get(
                "auto_open_time"),
            "auto_close_time" : status.get(
                "auto_close_time")
        })

    return jsonify({"success": True,
                    "courses": result})

@app.route("/api/lecturer/course/create",
           methods=["POST"])
def create_course():
    """
    Create course using session auth.
    No password re-entry needed — session is active.
    """
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get("course_code", "").upper()
    course_title= data.get("course_title", "")
    venue       = data.get("venue", "")
    start_time  = data.get("start_time", "")
    end_time    = data.get("end_time", "")
    day         = data.get("day", "")

    # Validate session
    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated. "
                       "Please log in again."
        }), 401

    if not all([course_code, course_title,
                venue, start_time, end_time, day]):
        return jsonify({
            "success": False,
            "message": "All course fields required"
        }), 400

    course_id = db.create_course(
        course_code  = course_code,
        course_title = course_title,
        venue        = venue,
        start_time   = start_time,
        end_time     = end_time,
        day          = day,
        lecturer_id  = session["lecturer_id"]
    )

    if course_id:
        return jsonify({
            "success"    : True,
            "message"    : f"Course {course_code} created!",
            "course_id"  : course_id,
            "course_code": course_code
        })
    return jsonify({
        "success": False,
        "message": f"Course {course_code} already exists"
    }), 409

@app.route("/api/lecturer/course/edit",
           methods=["PUT"])
def edit_course():
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get("course_code", "").upper()

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    # Verify ownership
    course = db.get_course(course_code)
    if not course:
        return jsonify({
            "success": False,
            "message": "Course not found"
        }), 404

    if course[7] != session["lecturer_id"]:
        return jsonify({
            "success": False,
            "message": "Access denied"
        }), 403

    db.update_course(
        course_code  = course_code,
        course_title = data.get("course_title"),
        venue        = data.get("venue"),
        start_time   = data.get("start_time"),
        end_time     = data.get("end_time"),
        day          = data.get("day")
    )

    return jsonify({
        "success": True,
        "message": "Course updated!"
    })

@app.route("/api/lecturer/course/edit-details",
           methods=["PUT"])
def edit_course_details():
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get("course_code", "").upper()

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    course = db.get_course(course_code)
    if not course:
        return jsonify({
            "success": False,
            "message": "Course not found"
        }), 404

    if course[7] != session["lecturer_id"]:
        return jsonify({
            "success": False,
            "message": "Access denied"
        }), 403

    db.update_course(
        course_code  = course_code,
        course_title = data.get("course_title"),
        venue        = data.get("venue"),
        start_time   = data.get("start_time"),
        end_time     = data.get("end_time"),
        day          = data.get("day")
    )

    return jsonify({
        "success": True,
        "message": "Course updated!"
    })

@app.route("/api/lecturer/course/delete",
           methods=["DELETE"])
def delete_course():
    """Delete a course (owner only)."""
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    success, message = db.delete_course(
        course_code, session["lecturer_id"])
    return jsonify({"success": success,
                    "message": message})

# ─────────────────────────────────────────────
# SESSION CONTROL ENDPOINTS
# ─────────────────────────────────────────────
def _session_control_action(action_name):
    """Generic session control using session auth."""
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    # Set admin current_lecturer from session
    admin.current_lecturer = {
        "id"  : session["lecturer_id"],
        "name": "Lecturer",
        "role": "lecturer"
    }

    actions = {
        "start_session"  : admin.start_session,
        "end_session"    : admin.end_session,
        "open_att"       : admin.manual_open_attendance,
        "close_att"      : admin.manual_close_attendance,
        "open_fb"        : admin.manual_open_feedback,
        "close_fb"       : admin.manual_close_feedback,
    }

    fn = actions.get(action_name)
    if not fn:
        return jsonify({
            "success": False,
            "message": "Unknown action"
        }), 400

    result = fn(course_code)
    admin.current_lecturer = None

    return jsonify({
        "success": bool(result),
        "message": "Done" if result else "Action failed"
    })

@app.route("/api/lecturer/session/start",
           methods=["POST"])
def start_session():
    return _session_control_action("start_session")

@app.route("/api/lecturer/session/end",
           methods=["POST"])
def end_session():
    return _session_control_action("end_session")

@app.route("/api/lecturer/control/open-att",
           methods=["POST"])
def open_attendance():
    return _session_control_action("open_att")

@app.route("/api/lecturer/control/close-att",
           methods=["POST"])
def close_attendance():
    return _session_control_action("close_att")

@app.route("/api/lecturer/control/open-fb",
           methods=["POST"])
def open_feedback():
    return _session_control_action("open_fb")

@app.route("/api/lecturer/control/close-fb",
           methods=["POST"])
def close_feedback():
    return _session_control_action("close_fb")

# ─────────────────────────────────────────────
# REPORT ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/reports/attendance",
           methods=["GET"])
def get_attendance_report():
    user_id     = request.args.get("user_id")
    course_code = request.args.get(
        "course_code", "").upper()
    date        = request.args.get("date")

    # Verify session
    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    # Verify ownership
    course = db.get_course(course_code)
    if not course:
        return jsonify({
            "success": False,
            "message": "Course not found"
        }), 404

    if course[7] != session["lecturer_id"]:
        return jsonify({
            "success": False,
            "message": "Access denied"
        }), 403

    records = db.get_attendance(
        course_code=course_code, date=date)

    present  = sum(1 for r in records
                   if r[12] == "PRESENT")
    rejected = len(records) - present
    total    = len(records)
    rate     = round(present/total*100, 1) \
               if total > 0 else 0

    result = []
    for r in records:
        result.append({
            "student_id"    : r[1],
            "status"        : r[12],
            "confidence"    : r[5],
            "spoof_verdict" : r[6],
            "location"      : r[7],
            "feedback_given": bool(r[8]),
            "feedback_text" : r[9],
            "sentiment"     : r[10],
            "timestamp"     : r[14]
        })

    return jsonify({
        "success": True,
        "summary": {
            "total"   : total,
            "present" : present,
            "rejected": rejected,
            "rate"    : rate
        },
        "records": result
    })

@app.route("/api/reports/sentiment",
           methods=["GET"])
def get_sentiment_report():
    course_code = request.args.get(
        "course_code", "").upper()
    date        = request.args.get("date")

    conn   = _get_db_conn()
    cursor = conn.cursor()
    query  = """
        SELECT sentiment, COUNT(*) as cnt,
               AVG(sentiment_conf) as avg_c
        FROM attendance
        WHERE course_code = ?
          AND sentiment IS NOT NULL
          AND feedback_given = 1
    """
    params = [course_code]
    if date:
        query  += " AND timestamp LIKE ?"
        params.append(f"{date}%")
    query += " GROUP BY sentiment"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    summary = {
        "Positive": {"count": 0, "avg_conf": 0},
        "Negative": {"count": 0, "avg_conf": 0},
        "Neutral" : {"count": 0, "avg_conf": 0}
    }
    for r in rows:
        if r[0] in summary:
            summary[r[0]] = {
                "count"   : r[1],
                "avg_conf": round(r[2] or 0, 2)
            }

    return jsonify({"success": True,
                    "summary": summary})

@app.route("/api/reports/export", methods=["GET"])
def export_report():
    user_id     = request.args.get("user_id")
    course_code = request.args.get(
        "course_code", "").upper()
    date        = request.args.get(
        "date", datetime.now().strftime("%Y-%m-%d"))

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    # Set admin session
    admin.current_lecturer = {
        "id"  : session["lecturer_id"],
        "name": "Lecturer",
        "role": "lecturer"
    }
    path = admin.export_report(course_code, date)
    admin.current_lecturer = None

    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return jsonify({"success": True,
                        "report" : data})

    return jsonify({
        "success": False,
        "message": "Export failed"
    }), 500

# ─────────────────────────────────────────────
# SYSTEM ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/system/health", methods=["GET"])
def system_health():
    logger.warning("🔥 HEALTH ROUTE CALLED!")
    return jsonify({
        "success"  : True,
        "status"   : "running",
        "timestamp": datetime.now().isoformat(),
        "version"  : "1.0.0",
        "models"   : {
            "voice_recognition" : "97.43%",
            "sentiment_analysis": "97.23%"
        },
        "database": "SQLite (active)"
    })

@app.route("/api/system/status", methods=["GET"])
def system_status():
    courses = db.get_courses()
    result  = []
    for c in courses:
        status = db.get_session_status(c[1])
        result.append({
            "course_code"     : c[1],
            "course_title"    : c[2],
            "attendance_open" : status["attendance_open"],
            "feedback_open"   : status["feedback_open"]
        })
    return jsonify({"success": True,
                    "courses": result})

@app.route("/reset-password")
def reset_password_page():
    return send_from_directory("frontend",
                                "index.html")

# ─────────────────────────────────────────────
# EMAIL VERIFICATION ENDPOINTS
# ─────────────────────────────────────────────

def _generate_code():
    """Generate 6-digit verification code."""
    import random
    return str(random.randint(100000, 999999))

def _send_verification_code(email, code, name):
    """
    Send email verification code.
    Falls back to console in dev mode.
    """
    print(f"\n  📧 EMAIL VERIFICATION CODE")
    print(f"  ─────────────────────────────────")
    print(f"  To   : {email}")
    print(f"  Name : {name}")
    print(f"  Code : {code}")
    print(f"  ─────────────────────────────────")

    if not EMAIL_CONFIG["ENABLED"]:
        print(f"  ℹ️  Email disabled — use code above")
        return True

    try:
        body = (
            f"Hello {name},\n\n"
            f"Your email verification code is:\n\n"
            f"  {code}\n\n"
            f"This code expires in 10 minutes.\n\n"
            f"If you did not request this, "
            f"ignore this email.\n\n"
            f"— {EMAIL_CONFIG['APP_NAME']}"
        )
        msg            = MIMEMultipart()
        msg["From"]    = EMAIL_CONFIG["SENDER"]
        msg["To"]      = email
        msg["Subject"] = (
            f"Email Verification — "
            f"{EMAIL_CONFIG['APP_NAME']}")
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(
                EMAIL_CONFIG["HOST"],
                EMAIL_CONFIG["PORT"]) as s:
            s.starttls()
            s.login(EMAIL_CONFIG["SENDER"],
                    EMAIL_CONFIG["PASSWORD"])
            s.sendmail(EMAIL_CONFIG["SENDER"],
                       email, msg.as_string())
        return True
    except Exception as e:
        print(f"  ❌ Email error: {e}")
        return False


@app.route("/api/profile/request-email-change",
           methods=["POST"])
def request_email_change():
    """
    Step 1: Request email update.
    Validates format, generates code,
    sends to NEW email address.
    """
    import re
    data      = request.get_json()
    user_id   = data.get("user_id")
    role      = data.get("role")
    new_email = data.get("new_email","").strip().lower()
    name      = data.get("name", "User")

    if not all([user_id, role, new_email]):
        return jsonify({
            "success": False,
            "message": "user_id, role and "
                       "new_email required"
        }), 400

    # Format validation
    pattern = (r'^[a-zA-Z0-9._%+\-]+'
               r'@[a-zA-Z0-9.\-]+'
               r'\.[a-zA-Z]{2,}$')
    if not re.match(pattern, new_email):
        return jsonify({
            "success": False,
            "message": "Invalid email format"
        }), 400

    # Check not already in use
    exists, _ = db.email_exists(new_email)
    if exists:
        return jsonify({
            "success": False,
            "message": "Email already in use "
                       "by another account"
        }), 409

    # Generate code and store
    code       = _generate_code()
    expires_at = (datetime.now() +
                  timedelta(minutes=10)).isoformat()

    db.store_email_verification_code(
        user_id, role, new_email, code, expires_at)

    # Send code to the NEW email
    _send_verification_code(new_email, code, name)

    return jsonify({
        "success"  : True,
        "message"  : (
            "Verification code sent to "
            f"{new_email}"),
        "dev_code" : code,   # remove in production
        "new_email": new_email
    })


@app.route("/api/profile/verify-email-code",
           methods=["POST"])
def verify_email_code():
    """
    Step 2: Verify code and update email.
    """
    data    = request.get_json()
    user_id = data.get("user_id")
    code    = data.get("code", "").strip()

    if not user_id or not code:
        return jsonify({
            "success": False,
            "message": "user_id and code required"
        }), 400

    success, message, new_email, role = (
        db.verify_email_code(user_id, code))

    if not success:
        return jsonify({
            "success": False,
            "message": message
        }), 400

    # Update email in database
    ok, msg = db.update_email(user_id, role, new_email)

    if ok:
        # Update session
        session = _get_session(user_id)
        if session:
            session["email"] = new_email
            _set_session(user_id, session)

    return jsonify({
        "success"  : ok,
        "message"  : msg if ok else message,
        "new_email": new_email if ok else None
    })


# ─────────────────────────────────────────────
# STUDENT COURSE EDIT ENDPOINT
# ─────────────────────────────────────────────
@app.route("/api/student/course/edit",
           methods=["PUT"])
def student_edit_course():
    """
    Student edits venue/time of an enrolled course.
    Venues restricted to:
      - CS Hardware Lab
      - CS Software Lab
    """
    data        = request.get_json()
    student_id  = data.get("student_id")
    course_code = data.get(
        "course_code", "").upper().strip()
    venue       = data.get("venue")
    start_time  = data.get("start_time")
    end_time    = data.get("end_time")

    ALLOWED_VENUES = [
        "CS Hardware Lab",
        "CS Software Lab"
    ]

    if venue and venue not in ALLOWED_VENUES:
        return jsonify({
            "success": False,
            "message": (
                "Venue must be either "
                "'CS Hardware Lab' or "
                "'CS Software Lab'")
        }), 400

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and "
                       "course_code required"
        }), 400

    success, message = db.update_course_details(
        course_code = course_code,
        venue       = venue,
        start_time  = start_time,
        end_time    = end_time
    )

    return jsonify({"success": success,
                    "message": message})

# ─────────────────────────────────────────────
# VOICE ENROLLMENT ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/enrollment/status",
           methods=["GET"])
def enrollment_status():
    """
    Check if student has enrolled their voice.
    Called after login to decide whether to
    show enrollment prompt.
    """
    student_id = request.args.get("student_id")
    if not student_id:
        return jsonify({
            "success": False,
            "message": "student_id required"
        }), 400

    status = db.get_enrollment_status(student_id)
    return jsonify({
        "success" : True,
        "enrolled": status["enrolled"],
        "details" : status
    })


@app.route("/api/enrollment/record",
           methods=["POST"])
def record_enrollment_sample():
    """
    Process a single voice enrollment recording.
    Handles WebM (browser) → WAV conversion.
    Preprocesses audio and extracts MFCC features.
    """
    student_id    = request.form.get("student_id")
    sample_number = int(request.form.get(
        "sample_number", 1))
    audio_file    = request.files.get("audio")

    if not student_id or not audio_file:
        return jsonify({
            "success": False,
            "message": "student_id and audio required"
        }), 400

    # ── Save uploaded file (WebM from browser)
    tmp_webm = tempfile.NamedTemporaryFile(
        suffix=".webm", delete=False)
    audio_file.save(tmp_webm.name)
    tmp_webm.close()

    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False)
    tmp_wav.close()

    try:
        # ── Convert WebM → WAV
        wav_path = tmp_webm.name  # start with original
        try:
            from pydub import AudioSegment
            audio_seg = AudioSegment.from_file(
                tmp_webm.name)
            audio_seg = audio_seg.set_frame_rate(16000)
            audio_seg = audio_seg.set_channels(1)
            audio_seg.export(tmp_wav.name,
                             format="wav")
            wav_path  = tmp_wav.name
            print(f"  ✅ Converted WebM→WAV: "
                  f"{len(audio_seg)/1000:.2f}s")

        except ImportError:
            print("  ⚠️  pydub not installed")
            print("  💡 Run: pip install pydub")
            return jsonify({
                "success": False,
                "message": "Server audio conversion "
                           "not configured. "
                           "Contact administrator."
            }), 500

        except Exception as conv_err:
            print(f"  ⚠️  Conversion error: {conv_err}")
            # Last resort: try soundfile direct read
            try:
                import soundfile as sf
                import numpy as np
                data, srate = sf.read(tmp_webm.name)
                if data.ndim > 1:
                    data = data[:, 0]
                sf.write(tmp_wav.name, data,
                          srate,
                          subtype='PCM_16')
                wav_path = tmp_wav.name
                print("  ✅ soundfile fallback worked")
            except Exception:
                print("  ❌ All conversions failed")
                return jsonify({
                    "success": False,
                    "message": "Could not process audio "
                               "format. Please ensure "
                               "FFmpeg is installed."
                }), 500

        # ── Preprocess audio
        from src.preprocessing.preprocess import (
            preprocess_audio, extract_mfcc)

        audio, sr = preprocess_audio(wav_path)

        if audio is None or len(audio) == 0:
            return jsonify({
                "success": False,
                "message": "Could not process audio. "
                           "Please speak clearly "
                           "and try again."
            }), 400

        # Quality check — must not be silent
        import numpy as np
        rms = np.sqrt(np.mean(audio ** 2))
        print(f"  📊 RMS level: {rms:.4f}")

        if rms < 0.001:
            return jsonify({
                "success": False,
                "message": "Audio too quiet. "
                           "Speak louder and "
                           "closer to the mic."
            }), 400

        # ── Extract MFCC features
        mfcc = extract_mfcc(audio, sr)
        print(f"  ✅ MFCC extracted: {mfcc.shape}")

        # ── Store features in session
        session = _get_session(student_id)
        if not session:
            session = {
                "email"           : "",
                "role"            : "student",
                "lecturer_id"     : None,
                "enrollment_feats": []
            }

        if "enrollment_feats" not in session:
            session["enrollment_feats"] = []

        session["enrollment_feats"].append(
            mfcc.tolist())
        _set_session(student_id, session)

        current_count = len(
            session["enrollment_feats"])
        print(f"  ✅ Sample {current_count} stored "
              f"for {student_id}")

        return jsonify({
            "success"      : True,
            "sample_number": sample_number,
            "total_so_far" : current_count,
            "message"      : (
                f"Sample {sample_number} "
                f"recorded ✅"),
            "features_shape": list(mfcc.shape),
            "duration"     : round(
                len(audio) / sr, 2),
            "rms"          : round(float(rms), 4)
        })

    except Exception as e:
        import traceback
        print(f"  ❌ Enrollment error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Processing failed: {str(e)}"
        }), 500

    finally:
        # Clean up temp files
        for tmp in [tmp_webm.name, tmp_wav.name]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


@app.route("/api/enrollment/save",
           methods=["POST"])
def save_enrollment():
    """
    Save all collected voice samples to database.
    Called after student completes all recordings.

    Retrieves features from session and saves
    to voice_templates table.
    """
    import numpy as np

    data       = request.get_json()
    student_id = data.get("student_id")

    if not student_id:
        return jsonify({
            "success": False,
            "message": "student_id required"
        }), 400

    session = _get_session(student_id)
    if not session:
        return jsonify({
            "success": False,
            "message": "Session expired. "
                       "Please log in again."
        }), 401

    feats = session.get("enrollment_feats", [])

    if len(feats) < 7:
        return jsonify({
            "success": False,
            "message": f"Not enough samples. "
                       f"Recorded {len(feats)}/7 "
                       f"required."
        }), 400

    try:
        import json
        features_list = [
            np.array(f) for f in feats
        ]

        success = db.save_voice_enrollment(
            student_id    = student_id,
            features_list = features_list,
            sample_count  = len(features_list)
        )

        # Clear enrollment features from session
        session.pop("enrollment_feats", None)
        _set_session(student_id, session)

        return jsonify({
            "success"     : success,
            "message"     : (
                f"Voice enrolled successfully! "
                f"{len(features_list)} samples saved."),
            "sample_count": len(features_list)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Save failed: {str(e)}"
        }), 500


@app.route("/api/enrollment/reset",
           methods=["DELETE"])
def reset_enrollment():
    """
    Reset enrollment session (start over).
    Clears temporary features from session.
    """
    data       = request.get_json()
    student_id = data.get("student_id")

    session = _get_session(student_id)
    if session:
        session["enrollment_feats"] = []
        _set_session(student_id, session)

    return jsonify({
        "success": True,
        "message": "Enrollment reset. "
                   "You can start over."
    })

# ─────────────────────────────────────────────
# QR CODE ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/qr/generate", methods=["POST"])
def generate_qr():
    """
    Generate a dynamic QR code for attendance.
    Called by lecturer when attendance is open.
    QR expires every 30 seconds automatically.
    """
    data        = request.get_json()
    user_id     = data.get("user_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    session = _get_session(user_id)
    if not session or session["role"] != "lecturer":
        return jsonify({
            "success": False,
            "message": "Not authenticated"
        }), 401

    # Verify ownership
    course = db.get_course(course_code)
    if not course or course[7] != session["lecturer_id"]:
        return jsonify({
            "success": False,
            "message": "Access denied"
        }), 403

    qr_data = db.create_qr_session(
        course_code     = course_code,
        lecturer_id     = session["lecturer_id"],
        expires_seconds = QR_EXPIRY_SECONDS
    )

    return jsonify({
        "success" : True,
        "qr"      : qr_data,
        "demo_mode": DEMO_MODE,
        "expiry_seconds": QR_EXPIRY_SECONDS
    })


@app.route("/api/qr/current", methods=["GET"])
def get_current_qr():
    """Get the current active QR for a course."""
    course_code = request.args.get(
        "course_code", "").upper()
    qr = db.get_active_qr(course_code)

    if not qr:
        return jsonify({
            "success": False,
            "message": "No active QR"
        })

    now        = datetime.now()
    expires    = datetime.fromisoformat(
        qr["expires_at"])
    expires_in = max(0, int(
        (expires - now).total_seconds()))

    return jsonify({
        "success"   : True,
        "token"     : qr["token"],
        "expires_at": qr["expires_at"],
        "expires_in": expires_in
    })


# ─────────────────────────────────────────────
# WEB ATTENDANCE MARKING ENDPOINT
# ─────────────────────────────────────────────

@app.route("/api/attendance/mark-web",
           methods=["POST"])
def mark_attendance_web():
    """
    Mark attendance from web interface.

    Flow:
    1. System control check (is attendance open?)
       - If GPS valid → mark PRESENT → done
       - If GPS fails/inaccurate → go to QR
    2. QR verification (fallback)
       - Validate token + expiry + session
       - If valid → mark PRESENT → done
    3. If both fail → reject

    Note: Voice verification happens separately
    via the audio upload endpoint.
    This endpoint handles location verification
    and QR fallback.
    """
    data        = request.get_json()
    student_id  = data.get("student_id")
    course_code = data.get(
        "course_code", "").upper().strip()

    # Location data from device
    lat         = data.get("lat")
    lon         = data.get("lon")
    accuracy    = data.get("accuracy")  # meters

    # QR fallback
    qr_token    = data.get("qr_token")

    # Verify method used
    method_used = data.get("method", "GPS")

    if not student_id or not course_code:
        return jsonify({
            "success": False,
            "message": "student_id and course_code required"
        }), 400

    # ── Step 1: System Control Check
    att_status = db.get_session_status(course_code)
    if not att_status["attendance_open"]:
        return jsonify({
            "success": False,
            "message": "Attendance is currently CLOSED "
                       "for this course. Wait for "
                       "your lecturer to open it."
        }), 403

    # Get course venue
    course = db.get_course(course_code)
    if not course:
        return jsonify({
            "success": False,
            "message": "Course not found"
        }), 404

    venue_name = course[3]  # course venue

    # ── Step 2: GPS Verification (Primary)
    gps_result = None
    if lat is not None and lon is not None:
        from src.preprocessing.location_verification \
            import verify_location
        gps_result = verify_location(
            float(lat), float(lon),
            venue_name = venue_name,
            accuracy_m = float(accuracy)
                         if accuracy else None
        )

        if gps_result["allowed"]:
            # GPS passed → mark attendance
            record_id = db.log_attendance_web(
                student_id       = student_id,
                course_code      = course_code,
                location_verdict = "ALLOWED",
                location_method  = "GPS",
                status           = "LOCATION_VERIFIED",
                spoof_verdict    = "PENDING"
            )

            print(f"  ✅ GPS verified: {student_id} "
                  f"at {venue_name} "
                  f"({gps_result['distance_m']}m)")

            return jsonify({
                "success"        : True,
                "method"         : "GPS",
                "message"        : gps_result["message"],
                "distance_m"     : gps_result["distance_m"],
                "radius_m"       : gps_result["radius_m"],
                "venue"          : venue_name,
                "next_step"      : "voice_capture",
                "record_id"      : record_id
            })

        # GPS failed — check if unreliable
        if gps_result["verdict"] == "GPS_UNRELIABLE":
            reason = "gps_unreliable"
        else:
            reason = "outside_geofence"

    else:
        reason = "no_gps"
        gps_result = {
            "verdict": "NO_GPS",
            "message": "GPS not available"
        }

    # ── Step 3: QR Fallback
    if qr_token:
        qr_ok, qr_msg = db.verify_qr_token(
            token        = qr_token,
            course_code  = course_code,
            student_id   = student_id
        )

        if qr_ok:
            record_id = db.log_attendance_web(
                student_id       = student_id,
                course_code      = course_code,
                location_verdict = "QR_VERIFIED",
                location_method  = "QR",
                status           = "LOCATION_VERIFIED",
                spoof_verdict    = "PENDING"
            )

            print(f"  ✅ QR verified: {student_id} "
                  f"for {course_code}")

            return jsonify({
                "success"  : True,
                "method"   : "QR",
                "message"  : "QR code verified ✅",
                "next_step": "voice_capture",
                "record_id": record_id
            })
        else:
            return jsonify({
                "success"   : False,
                "method"    : "QR",
                "message"   : qr_msg,
                "gps_reason": reason
            }), 400

    # ── Both failed — return GPS failure
    # with instruction to use QR
    return jsonify({
        "success"      : False,
        "method"       : "GPS",
        "gps_verdict"  : gps_result.get("verdict"),
        "gps_reason"   : reason,
        "message"      : gps_result.get("message",
            "Location verification failed."),
        "fallback"     : "QR",
        "fallback_msg" : "GPS failed. Please scan "
                         "the QR code shown by your "
                         "lecturer.",
        "distance_m"   : gps_result.get("distance_m"),
        "radius_m"     : gps_result.get("radius_m")
    }), 400


@app.route("/api/attendance/complete",
           methods=["POST"])
def complete_attendance():
    """
    Complete attendance after voice verification.
    Updates the attendance record with voice result.
    Called after voice capture succeeds.
    """
    data       = request.get_json()
    record_id  = data.get("record_id")
    student_id = data.get("student_id")
    confidence = data.get("confidence", 0)
    verified   = data.get("verified", False)

    if not record_id or not student_id:
        return jsonify({
            "success": False,
            "message": "record_id and student_id required"
        }), 400

    conn   = _get_db_conn()
    cursor = conn.cursor()

    if verified:
        cursor.execute("""
            UPDATE attendance
            SET status        = 'PRESENT',
                confidence    = ?,
                spoof_verdict = 'LIVE'
            WHERE id         = ?
              AND student_id = ?
        """, (confidence, record_id, student_id))
        status  = "PRESENT"
        message = "✅ Attendance marked successfully!"
    else:
        cursor.execute("""
            UPDATE attendance
            SET status        = 'REJECTED',
                confidence    = ?,
                spoof_verdict = 'FAILED'
            WHERE id         = ?
              AND student_id = ?
        """, (confidence, record_id, student_id))
        status  = "REJECTED"
        message = ("❌ Voice not recognized. "
                   "Attendance rejected.")

    conn.commit()
    conn.close()

    return jsonify({
        "success"   : verified,
        "status"    : status,
        "confidence": confidence,
        "message"   : message
    })


# ─────────────────────────────────────────────
# UPDATE COURSES WITH CORRECT VENUES
# ─────────────────────────────────────────────
@app.route("/api/admin/fix-venues",
           methods=["POST"])
def fix_course_venues():
    """
    One-time fix: update all seeded courses
    to use only CS Hardware Lab / CS Software Lab.
    """
    conn   = _get_db_conn()
    cursor = conn.cursor()

    updates = [
        ("CS Hardware Lab",
         "08:00", "10:00", "CSC308"),
        ("CS Software Lab",
         "10:00", "12:00", "CSC306"),
        ("CS Hardware Lab",
         "12:00", "14:00", "CSC318"),
        ("CS Software Lab",
         "14:00", "16:00", "CSC320"),
        ("CS Hardware Lab",
         "08:00", "10:00", "CSC322"),
    ]

    for venue, start, end, code in updates:
        cursor.execute("""
            UPDATE courses
            SET venue      = ?,
                start_time = ?,
                end_time   = ?,
                updated_at = ?
            WHERE course_code = ?
        """, (venue, start, end,
               datetime.now().isoformat(), code))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "Course venues updated"
    })

@app.route("/api/admin/reset-system-data",
           methods=["POST"])
def reset_system_data_endpoint():
    """Reset test-generated browser data back to the seeded demo state."""
    payload = request.get_json(silent=True) or {}
    secret  = (
        payload.get("secret") or
        request.headers.get("X-RESET-SECRET", "")
    )

    if secret != RESET_SECRET:
        return jsonify({
            "success": False,
            "message": "Invalid reset secret"
        }), 403

    try:
        success, message = db.reset_system_data()
        return jsonify({
            "success": success,
            "message": message
        })
    except Exception as exc:
        return jsonify({
            "success": False,
            "message": str(exc)
        }), 500

@app.route("/api/attendance/check-auto",
           methods=["POST"])
def check_auto_attendance():
    """
    Trigger auto-open check for a course.
    Called by frontend when student opens
    course page — ensures auto-open fires
    even between 60-second intervals.
    """
    data        = request.get_json() or {}
    course_code = data.get(
        "course_code", "").upper()

    if not course_code:
        return jsonify({
            "success": False
        }), 400

    # Run both checks immediately
    _auto_open_by_course_time()
    _auto_check_all_sessions()

    status = db.get_session_status(course_code)
    course = db.get_course(course_code)

    return jsonify({
        "success"         : True,
        "course_code"     : course_code,
        "attendance_open" : status[
            "attendance_open"],
        "feedback_open"   : status["feedback_open"],
        "auto_open_time"  : status.get(
            "auto_open_time"),
        "auto_close_time" : status.get(
            "auto_close_time"),
        "venue"           : course[3]
                             if course else None
    })

@app.route("/api/attendance/history",
           methods=["GET"])
def attendance_history():
    """
    Get attendance history for a student.
    Returns all records with course, status,
    confidence, sentiment, timestamp.
    """
    student_id  = request.args.get("student_id")
    course_code = request.args.get("course_code")
    date        = request.args.get("date")

    if not student_id:
        return jsonify({
            "success": False,
            "message": "student_id required"
        }), 400

    try:
        records = db.get_attendance(
            student_id  = student_id,
            course_code = course_code,
            date        = date
        )

        result = []
        for r in records:
            result.append({
                "id"              : r[0],
                "student_id"      : r[1],
                "course_code"     : r[3],
                "course_title"    : r[4],
                "confidence"      : r[5],
                "spoof_verdict"   : r[6],
                "location_verdict": r[7],
                "feedback_given"  : bool(r[8]),
                "feedback_text"   : r[9],
                "sentiment"       : r[10],
                "sentiment_conf"  : r[11],
                "status"          : r[12],
                "timestamp"       : r[14]
            })

        return jsonify({
            "success": True,
            "records": result,
            "total"  : len(result)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Could not load history: "
                       f"{str(e)}",
            "records": [],
            "total"  : 0
        }), 500
    
# ─────────────────────────────────────────────
# SIGNUP EMAIL VERIFICATION
# ─────────────────────────────────────────────

def _send_signup_verification_code(email,
                                    code,
                                    name):
    """
    Send signup verification email.
    Shows code in terminal for dev mode.
    """
    print(f"\n  📧 SIGNUP VERIFICATION CODE")
    print(f"  ─────────────────────────────────")
    print(f"  To   : {email}")
    print(f"  Name : {name}")
    print(f"  Code : {code}")
    print(f"  ─────────────────────────────────")

    if not EMAIL_CONFIG.get("ENABLED"):
        print("  ℹ️  Email disabled — use code above")
        return True

    try:
        body = (
            f"Hello {name},\n\n"
            f"Welcome to the Voice Attendance &\n"
            f"Sentiment Feedback System — ABUAD.\n\n"
            f"Your email verification code is:\n\n"
            f"        {code}\n\n"
            f"Enter this code to complete your\n"
            f"registration. It expires in 10 minutes.\n\n"
            f"If you did not sign up, ignore this.\n\n"
            f"— CS Department, ABUAD"
        )
        msg            = MIMEMultipart()
        msg["From"]    = EMAIL_CONFIG["SENDER"]
        msg["To"]      = email
        msg["Subject"] = (
            "Verify Your Email — VASFS ABUAD")
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(
                EMAIL_CONFIG["HOST"],
                EMAIL_CONFIG["PORT"]) as s:
            s.starttls()
            s.login(EMAIL_CONFIG["SENDER"],
                    EMAIL_CONFIG["PASSWORD"])
            s.sendmail(
                EMAIL_CONFIG["SENDER"],
                email, msg.as_string())
        print(f"  ✅ Verification email sent!")
        return True
    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        return False


@app.route("/api/auth/signup/send-code",
           methods=["POST"])
def signup_send_code():
    """
    Step 1 of student registration.
    Validates all fields THEN sends verification
    code to email. Account NOT created yet.
    """
    import re
    import random

    data     = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get(
        "email", "").strip().lower()
    matric   = data.get(
        "matric_number", "").strip().upper()
    password = data.get("password", "").strip()
    confirm  = data.get(
        "confirm_password", "").strip()

    # Presence check
    if not all([name, email, matric,
                password, confirm]):
        return jsonify({
            "success": False,
            "message": "All fields are required"
        }), 400

    # Password match
    if password != confirm:
        return jsonify({
            "success": False,
            "message": "Passwords do not match"
        }), 400

    # Email format
    pat = (r'^[a-zA-Z0-9._%+\-]+'
           r'@[a-zA-Z0-9.\-]+'
           r'\.[a-zA-Z]{2,}$')
    if not re.match(pat, email):
        return jsonify({
            "success": False,
            "message": "Invalid email format. "
                       "Use: name@gmail.com"
        }), 400

    # Matric format
    mat_pat = r'^\d{2}/SCI01/\d{3}$'
    if not re.match(mat_pat, matric):
        return jsonify({
            "success": False,
            "message": "Invalid matric format. "
                       "Use: 22/SCI01/114"
        }), 400

    # Password strength
    is_strong, msg, _ = \
        db.check_password_strength(password)
    if not is_strong:
        return jsonify({
            "success": False,
            "message": msg
        }), 400

    # Email uniqueness
    exists, _ = db.email_exists(email)
    if exists:
        return jsonify({
            "success": False,
            "message": "Email already registered. "
                       "Use a different email."
        }), 409

    # Matric uniqueness
    conn   = _get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT matric_number FROM students
        WHERE matric_number = ?
    """, (matric,))
    if cursor.fetchone():
        conn.close()
        return jsonify({
            "success": False,
            "message": "Matric number already "
                       "registered."
        }), 409
    conn.close()

    # Generate code and store pending data
    code       = str(random.randint(100000,
                                     999999))
    expires_at = (datetime.now() +
                  timedelta(minutes=10)).isoformat()

    pending = {
        "name"    : name,
        "email"   : email,
        "matric"  : matric,
        "password": password
    }

    db.store_signup_verification(
        email, code, expires_at, pending)

    # Send verification code
    _send_signup_verification_code(
        email, code, name)

    return jsonify({
        "success"  : True,
        "message"  : (f"Verification code sent "
                      f"to {email}"),
        "dev_code" : code,
        "email"    : email
    })


@app.route("/api/auth/signup/verify-code",
           methods=["POST"])
def signup_verify_code():
    """
    Step 2 of student registration.
    Verifies email code then creates account.
    """
    data  = request.get_json()
    email = data.get("email","").strip().lower()
    code  = data.get("code","").strip()

    if not email or not code:
        return jsonify({
            "success": False,
            "message": "Email and code required"
        }), 400

    ok, message, pending = db.verify_signup_code(
        email, code)

    if not ok:
        return jsonify({
            "success": False,
            "message": message
        }), 400

    # Create account now that email is verified
    success, reg_msg, student_id = \
        db.register_student(
            pending["name"],
            pending["email"],
            pending["matric"],
            pending["password"]
        )

    if not success:
        return jsonify({
            "success": False,
            "message": reg_msg
        }), 400

    # Auto-login
    user = db.login(email, pending["password"])
    if user:
        db.log_login(user["id"], user["role"])
        _set_session(user["id"], {
            "email"      : user["email"],
            "role"       : user["role"],
            "lecturer_id": None
        })

    print(f"  ✅ Account created: {student_id} "
          f"({pending['name']})")

    return jsonify({
        "success"   : True,
        "message"   : "Account created!",
        "student_id": student_id,
        "user"      : {
            "id"           : student_id,
            "name"         : pending["name"],
            "email"        : email,
            "role"         : "student",
            "matric_number": pending["matric"],
            "voice_enrolled": False
        }
    }), 201

# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import socket

    # Get local IP for display
    try:
        s = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    # Check SSL certificates exist
    use_https = (
        os.path.exists('cert.pem') and
        os.path.exists('key.pem')
    )

    print("\n" + "=" * 55)
    print("   🚀 Starting Flask Backend Server")
    print("=" * 55)

    if use_https:
        print(f"\n  🔒 HTTPS Mode (Location works"
              f" on all devices)")
        print(f"\n  Laptop : https://localhost:5000")
        print(f"  Phone  : https://{local_ip}:5000")
        print(f"\n  ⚠️  First visit: browser will warn"
              f" 'Not Safe'")
        print(f"     Click 'Advanced' → "
              f"'Proceed to site' → done")
        print(f"     Only needed ONCE per device")
    else:
        print(f"\n  ⚠️  HTTP Mode (no SSL cert found)")
        print(f"  Laptop : http://localhost:5000")
        print(f"  Phone  : http://{local_ip}:5000")
        print(f"\n  Run: python generate_cert.py")
        print(f"  to enable HTTPS for phone GPS")

    print(f"\n  Press CTRL+C to stop\n")
    print("=" * 55)

    # Start background scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _auto_check_all_sessions,
        'interval',
        seconds = 60,
        id      = 'auto_session_check'
    )
    scheduler.add_job(
        _auto_open_by_course_time,
        'interval',
        seconds = 60,
        id      = 'auto_time_check'
    )
    scheduler.start()
    print("  ✅ Background scheduler started")
    print("  ℹ️  Auto-open/close checks every 60s\n")

    server_port = int(os.environ.get("PORT", 5000))
    server_host = "0.0.0.0"

    if use_https:
        print("  ⚡ Using Flask HTTPS server with threading")
        app.run(
            host        = server_host,
            port        = server_port,
            debug       = False,
            ssl_context = ('cert.pem', 'key.pem'),
            threaded    = True,
            use_reloader= False
        )
    else:
        print("  ⚡ Using HTTP server")
        app.run(
            host        = server_host,
            port        = server_port,
            debug       = False,
            threaded    = True,
            use_reloader= False
        )
