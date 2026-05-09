import os
import sys
import json
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.abspath("."))
from src.modules.module8_database import DatabaseModule

# ─────────────────────────────────────────────
# MODULE 11: ADMINISTRATIVE (LECTURER) MODULE
# ─────────────────────────────────────────────
class AdminModule:
    """
    Administrative (Lecturer) Module.
    Uses DatabaseModule for ALL data operations.

    Features:
    A. Authentication & Access Control
    B. Course Management
    C. System Control (Hybrid Auto + Manual)
    D. Reporting & Dashboard
    E. Monitoring & Validation
    F. Session Management

    LECTURER LOGINS:
      peter@demo.com  / temp123  → CSC308
      anna@demo.com   / temp123  → CSC306
      james@demo.com  / temp123  → CSC318
      grace@demo.com  / temp123  → CSC320
      david@demo.com  / temp123  → CSC322
    """

    PLOTS_DIR = "outputs/plots/admin"

    def __init__(self):
        os.makedirs(self.PLOTS_DIR, exist_ok=True)
        self.db               = DatabaseModule()
        self.current_lecturer = None

    # ─────────────────────────────────────────
    # A. AUTHENTICATION & ACCESS CONTROL
    # ─────────────────────────────────────────
    def login(self, email, password,
               test_mode=False):
        """
        Lecturer login with email + password.
        Role = 'lecturer' only.
        First-login triggers password change prompt.
        """
        print("\n" + "─" * 50)
        print("  🔐 LECTURER LOGIN")
        print("─" * 50)

        if test_mode:
            print(f"  ℹ️  Test mode: {email}")

        user = self.db.login(email, password)

        if not user:
            print("  ❌ Invalid email or password")
            return False

        if user["role"] != "lecturer":
            print("  ❌ Access denied — "
                  "lecturer accounts only")
            return False

        self.current_lecturer = user
        print(f"  ✅ Welcome, {user['name']}!")
        print(f"     Role  : Lecturer")
        print(f"     ID    : {user['id']}")

        if user.get("first_login") and not test_mode:
            print(f"\n  ⚠️  First login detected!")
            print(f"     Please change your password.")
            self._change_password_prompt()

        return True

    def _change_password_prompt(self):
        """Prompt lecturer to change temporary password."""
        new_pw = input(
            "  Enter new password: ").strip()
        if len(new_pw) < 6:
            print("  ⚠️  Too short (min 6 chars)")
            return
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE lecturers
            SET password = ?, first_login = 0
            WHERE lecturer_id = ?
        """, (self.db._hash(new_pw),
               self.current_lecturer["id"]))
        conn.commit()
        conn.close()
        print("  ✅ Password updated!")

    def logout(self):
        """Log out current lecturer."""
        if self.current_lecturer:
            print(f"\n  👋 Goodbye, "
                  f"{self.current_lecturer['name']}!")
        self.current_lecturer = None

    def _require_login(self):
        """Ensure a lecturer is logged in."""
        if not self.current_lecturer:
            print("  ❌ Please login first")
            return False
        return True

    def _require_ownership(self, course_code):
        """
        Verify logged-in lecturer owns this course.
        Lecturer can ONLY access courses where
        course.lecturer_id == logged_in_lecturer_id
        """
        if not self._require_login():
            return False
        course = self.db.get_course(course_code)
        if not course:
            print(f"  ❌ Course {course_code} not found")
            return False
        if course[6] != self.current_lecturer["id"]:
            print(f"  ❌ Access denied — "
                  f"{course_code} belongs to "
                  f"another lecturer")
            return False
        return True

    # ─────────────────────────────────────────
    # CORE SESSION HELPERS (CLEAN LOGIC)
    # ─────────────────────────────────────────
    def _set_session_state(self, course_code,
                            attendance_open=None,
                            feedback_open=None,
                            manually_overridden=None):
        """
        Update session state.
        
        Logic:
        - If manually_overridden=True: attendance/feedback 
          are manually set and will NOT be auto-changed
        - If manually_overridden=False: attendance/feedback 
          will follow the auto-open/close window
        """
        today   = datetime.now().strftime("%Y-%m-%d")
        conn    = sqlite3.connect(self.db.DB_PATH)
        cursor  = conn.cursor()

        cursor.execute("""
            SELECT id, attendance_open, feedback_open, 
                   manually_overridden
            FROM system_control
            WHERE course_code = ? AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code.upper(), today))
        row = cursor.fetchone()

        if row:
            # Read current values
            current_att = bool(row[1])
            current_fb  = bool(row[2])
            current_override = bool(row[3])

            # Only change what was explicitly provided
            new_att = attendance_open \
                      if attendance_open is not None \
                      else current_att
            new_fb  = feedback_open \
                      if feedback_open is not None \
                      else current_fb
            new_override = manually_overridden \
                           if manually_overridden is not None \
                           else current_override

            cursor.execute("""
                UPDATE system_control
                SET attendance_open     = ?,
                    feedback_open       = ?,
                    manually_overridden = ?
                WHERE id = ?
            """, (int(new_att), int(new_fb),
                   int(new_override), row[0]))
        else:
            # No session yet — create one
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
                 manually_overridden,
                 session_started_at, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course_id,
                course_code.upper(),
                self.current_lecturer["id"]
                    if self.current_lecturer else "",
                int(attendance_open or 0),
                int(feedback_open or 0),
                int(manually_overridden or 0),
                datetime.now().isoformat(),
                today
            ))

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # B. COURSE MANAGEMENT
    # ─────────────────────────────────────────
    def create_course(self, course_code,
                       course_title, venue,
                       start_time, end_time):
        """
        Create course — auto-owned by logged-in lecturer.
        Students join by entering the course_code.
        """
        if not self._require_login():
            return False

        course_id = self.db.create_course(
            course_code  = course_code,
            course_title = course_title,
            venue        = venue,
            start_time   = start_time,
            end_time     = end_time,
            lecturer_id  = self.current_lecturer["id"]
        )

        if course_id:
            print(f"\n  ✅ Course created!")
            print(f"     Code    : {course_code.upper()}")
            print(f"     Title   : {course_title}")
            print(f"     Venue   : {venue}")
            print(f"     Start   : {start_time}")
            print(f"     End     : {end_time}")
            print(f"     Owner   : "
                  f"{self.current_lecturer['name']}")
            print(f"\n  💡 Share '{course_code.upper()}' "
                  f"with students to enroll")
            return True
        else:
            print(f"  ⚠️  Course {course_code} "
                  f"already exists")
            return False

    def edit_course(self, course_code,
                     course_title=None,
                     venue=None,
                     start_time=None,
                     end_time=None):
        """
        Edit course details.
        Only the owning lecturer can edit.
        Only provided fields are updated.
        """
        if not self._require_ownership(course_code):
            return False

        self.db.update_course(
            course_code  = course_code,
            course_title = course_title,
            venue        = venue,
            start_time   = start_time,
            end_time     = end_time
        )

        print(f"  ✅ {course_code.upper()} updated:")
        if course_title:
            print(f"     Title  → {course_title}")
        if venue:
            print(f"     Venue  → {venue}")
        if start_time:
            print(f"     Start  → {start_time}")
        if end_time:
            print(f"     End    → {end_time}")
        return True

    def view_my_courses(self):
        """
        View all courses owned by logged-in lecturer.
        Shows attendance/feedback open status.
        """
        if not self._require_login():
            return []

        courses = self.db.get_courses(
            lecturer_id=self.current_lecturer["id"])

        print("\n" + "=" * 65)
        print(f"  📚 MY COURSES — "
              f"{self.current_lecturer['name']}")
        print("=" * 65)

        if not courses:
            print("  No courses yet.")
            print("  Use create_course() to add one.")
            return []

        print(f"  {'#':<4} {'Code':<10} "
              f"{'Title':<28} "
              f"{'Att':<6} {'FB':<6} {'Time'}")
        print(f"  {'─'*65}")

        for idx, c in enumerate(courses, 1):
            status = self.db.get_session_status(c[1])
            att    = "🟢" if status["attendance_open"] \
                     else "🔴"
            fb     = "🟢" if status["feedback_open"] \
                     else "🔴"
            time_s = f"{c[4]}-{c[5]}"
            print(f"  {idx:<4} {c[1]:<10} "
                  f"{c[2][:27]:<28} "
                  f"{att:<6} {fb:<6} {time_s}")

        print(f"\n  🟢 Open  🔴 Closed")
        print("=" * 65)
        return courses

    def select_course(self, test_mode=False,
                       test_course=None):
        """Select a course to manage."""
        courses = self.view_my_courses()
        if not courses:
            return None

        if test_mode and test_course:
            for c in courses:
                if c[1] == test_course.upper():
                    return {
                        "course_id"   : c[0],
                        "course_code" : c[1],
                        "course_title": c[2],
                        "venue"       : c[3],
                        "start_time"  : c[4],
                        "end_time"    : c[5],
                        "lecturer_id" : c[6]
                    }
            return None

        try:
            choice = int(input(
                f"\n  Select course "
                f"(1-{len(courses)}): "))
            if 1 <= choice <= len(courses):
                c = courses[choice - 1]
                return {
                    "course_id"   : c[0],
                    "course_code" : c[1],
                    "course_title": c[2],
                    "venue"       : c[3],
                    "start_time"  : c[4],
                    "end_time"    : c[5],
                    "lecturer_id" : c[6]
                }
        except (ValueError, IndexError):
            pass

        print("  ❌ Invalid selection")
        return None

    # ─────────────────────────────────────────
    # C. SYSTEM CONTROL (HYBRID AUTO + MANUAL)
    # ─────────────────────────────────────────
    def start_session(self, course_code,
                       test_mode=False):
        """
        F. Start lecture session.
        C. Sets hybrid auto time window:

        Both attendance AND feedback automatically
        open in the LAST 10 minutes of class,
        and both close at end_time.

        Example: class 10:00–12:00
          Auto-opens  : 11:50 (attendance + feedback)
          Auto-closes : 12:00 (attendance + feedback)

        Lecturer can override manually anytime.
        """
        if not self._require_ownership(course_code):
            return False

        course = self.db.get_course(course_code)
        if not course:
            print(f"  ❌ Course not found")
            return False

        today  = datetime.now().strftime("%Y-%m-%d")
        status = self.db.get_session_status(course_code)

        if status.get("started_at"):
            print(f"  ⚠️  Session already active "
                  f"for {course_code}")
            return False

        # Build auto times from course end_time
        try:
            end_str  = course[5]   # e.g. "12:00"
            end_dt   = datetime.strptime(
                f"{today} {end_str}",
                "%Y-%m-%d %H:%M")
            open_dt  = end_dt - timedelta(minutes=10)
        except Exception:
            now     = datetime.now()
            open_dt = now + timedelta(minutes=50)
            end_dt  = now + timedelta(minutes=60)

        # Create session record with auto times
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT course_id FROM courses
            WHERE course_code = ?
        """, (course_code.upper(),))
        cid       = cursor.fetchone()
        course_id = cid[0] if cid else None

        cursor.execute("""
            INSERT INTO system_control
            (course_id, course_code, lecturer_id,
             attendance_open, feedback_open,
             auto_open_time, auto_close_time,
             manually_overridden,
             session_started_at, date)
            VALUES (?, ?, ?, 0, 0, ?, ?, 0, ?, ?)
        """, (
            course_id,
            course_code.upper(),
            self.current_lecturer["id"],
            open_dt.isoformat(),
            end_dt.isoformat(),
            datetime.now().isoformat(),
            today
        ))
        conn.commit()
        conn.close()

        print(f"\n  ✅ Session STARTED — {course_code}")
        print(f"     Course      : {course[2]}")
        print(f"     Venue       : {course[3]}")
        print(f"     Schedule    : {course[4]}–{course[5]}")
        print(f"     Auto-opens  : "
              f"{open_dt.strftime('%H:%M')} "
              f"(attendance + feedback)")
        print(f"     Auto-closes : "
              f"{end_dt.strftime('%H:%M')}")
        print(f"\n  💡 Students can mark attendance "
              f"from {open_dt.strftime('%H:%M')}")
        return True

    def check_auto_open(self, course_code):
        """
        ✅ AUTOMATIC WINDOW LOGIC
        
        For class 10:00–12:00:
          Auto-opens  → 11:50 (last 10 min)
          Auto-closes → 12:00 (end of class)
        
        Rule:
        - If overridden=1: Do NOT auto-apply
          (lecturer's manual setting stays until they change it)
        - If overridden=0: Apply auto window logic
          (attendance/feedback follow the time window)
        """
        now    = datetime.now()
        status = self.db.get_session_status(course_code)

        # If manually overridden, leave it alone
        if status.get("overridden"):
            return False

        # Only proceed if we have auto times
        if not status.get("auto_open_time") or \
           not status.get("auto_close_time"):
            return False

        try:
            auto_open  = datetime.fromisoformat(
                status["auto_open_time"])
            auto_close = datetime.fromisoformat(
                status["auto_close_time"])
        except Exception:
            return False

        changed = False

        # ── AUTO-OPEN: If time >= 11:50 and not yet open
        if (now >= auto_open and
                not status["attendance_open"]):
            self._set_session_state(
                course_code,
                attendance_open    = True,
                feedback_open      = True,
                manually_overridden= False
            )
            print(f"  🟢 AUTO-OPENED: Attendance + "
                  f"Feedback for {course_code}")
            changed = True

        # ── AUTO-CLOSE: If time >= 12:00 and still open
        elif (now >= auto_close and
              status["attendance_open"]):
            self._set_session_state(
                course_code,
                attendance_open    = False,
                feedback_open      = False,
                manually_overridden= False
            )
            print(f"  🔴 AUTO-CLOSED: Attendance + "
                  f"Feedback for {course_code}")
            changed = True

        return changed

    def manual_open_attendance(self, course_code):
        """
        🔶 MANUAL OVERRIDE: Open attendance now.
        Sets manually_overridden=1 so auto-logic won't 
        interfere until lecturer manually closes it.
        """
        if not self._require_ownership(course_code):
            return False
        self._set_session_state(
            course_code,
            attendance_open    = True,
            manually_overridden= True
        )
        print(f"  🟢 MANUAL: Attendance OPENED "
              f"— {course_code}")
        return True

    def manual_close_attendance(self, course_code):
        """
        🔶 MANUAL OVERRIDE: Close attendance now.
        Sets manually_overridden=1 so auto-logic won't 
        interfere until lecturer manually opens it.
        """
        if not self._require_ownership(course_code):
            return False
        self._set_session_state(
            course_code,
            attendance_open    = False,
            manually_overridden= True
        )
        print(f"  🔴 MANUAL: Attendance CLOSED "
              f"— {course_code}")
        return True

    def manual_open_feedback(self, course_code):
        """
        🔶 MANUAL OVERRIDE: Open feedback now.
        Sets manually_overridden=1 so auto-logic won't 
        interfere until lecturer manually closes it.
        """
        if not self._require_ownership(course_code):
            return False
        self._set_session_state(
            course_code,
            feedback_open      = True,
            manually_overridden= True
        )
        print(f"  🟢 MANUAL: Feedback OPENED "
              f"— {course_code}")
        return True

    def manual_close_feedback(self, course_code):
        """
        🔶 MANUAL OVERRIDE: Close feedback now.
        Sets manually_overridden=1 so auto-logic won't 
        interfere until lecturer manually opens it.
        """
        if not self._require_ownership(course_code):
            return False
        self._set_session_state(
            course_code,
            feedback_open      = False,
            manually_overridden= True
        )
        print(f"  🔴 MANUAL: Feedback CLOSED "
              f"— {course_code}")
        return True

    def end_session(self, course_code):
        """
        🔴 END SESSION: Close both immediately.
        Marks manually_overridden=1 to prevent auto-reopen.
        """
        if not self._require_ownership(course_code):
            return False

        self._set_session_state(
            course_code,
            attendance_open    = False,
            feedback_open      = False,
            manually_overridden= True
        )

        # Mark session end time
        today  = datetime.now().strftime("%Y-%m-%d")
        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE system_control
            SET session_ended_at = ?
            WHERE course_code = ? AND date = ?
        """, (datetime.now().isoformat(),
               course_code.upper(), today))
        conn.commit()
        conn.close()

        print(f"\n  ✅ Session ENDED — {course_code}")
        print(f"     Attendance  : 🔴 CLOSED")
        print(f"     Feedback    : 🔴 CLOSED")
        return True

    # Student access checks
    def can_mark_attendance(self, course_code):
        """
        Student check: is attendance open?
        Triggers auto-open check first.
        """
        self.check_auto_open(course_code)
        return self.db.get_session_status(
            course_code)["attendance_open"]

    def can_give_feedback(self, course_code):
        """
        Student check: is feedback open?
        Triggers auto-open check first.
        """
        self.check_auto_open(course_code)
        return self.db.get_session_status(
            course_code)["feedback_open"]

    def display_course_page(self, course_code):
        """
        Display course info + session status.
        What the lecturer sees when they select a course.
        """
        if not self._require_ownership(course_code):
            return

        course = self.db.get_course(course_code)
        status = self.db.get_session_status(course_code)

        att = "🟢 OPEN" if status["attendance_open"] \
              else "🔴 CLOSED"
        fb  = "🟢 OPEN" if status["feedback_open"] \
              else "🔴 CLOSED"

        print("\n" + "=" * 55)
        print(f"  📖 COURSE: {course[1]}")
        print("=" * 55)
        print(f"  Title       : {course[2]}")
        print(f"  Venue       : {course[3]}")
        print(f"  Start Time  : {course[4]}")
        print(f"  End Time    : {course[5]}")
        print(f"  Lecturer    : "
              f"{self.current_lecturer['name']}")
        print("─" * 55)
        print(f"  Attendance  : {att}")
        print(f"  Feedback    : {fb}")
        if status.get("auto_open_time"):
            print(f"  Auto-opens  : "
                  f"{status['auto_open_time'][11:16]}")
            print(f"  Auto-closes : "
                  f"{status['auto_close_time'][11:16]}")
        print("─" * 55)
        print("  Controls:")
        print("  [1] Open Attendance manually")
        print("  [2] Close Attendance manually")
        print("  [3] Open Feedback manually")
        print("  [4] Close Feedback manually")
        print("  [5] End Session")
        print("  [6] View Reports")
        print("=" * 55)

    # ─────────────────────────────────────────
    # STUDENT ENROLLMENT
    # ─────────────────────────────────────────
    def enroll_student(self, student_id,
                        course_code):
        """Student joins course via course code."""
        ok, msg = self.db.enroll_student(
            student_id, course_code)
        if ok:
            print(f"  ✅ {student_id} enrolled "
                  f"in {course_code.upper()}")
        else:
            print(f"  ❌ {msg}")
        return ok

    # ─────────────────────────────────────────
    # D. REPORTING & DASHBOARD
    # ─────────────────────────────────────────
    def print_attendance_report(self, course_code,
                                  date=None):
        """
        D. Attendance Dashboard.
        List of students, present/absent, time marked.
        """
        if not self._require_ownership(course_code):
            return []

        records = self.db.get_attendance(
            course_code=course_code, date=date)
        today   = date or datetime.now().strftime(
            "%Y-%m-%d")

        print(f"\n  {'═'*62}")
        print(f"  ATTENDANCE — {course_code} | {today}")
        print(f"  {'═'*62}")
        print(f"  {'#':<4} {'Student':<14} "
              f"{'Status':<10} {'Conf':<8} "
              f"{'Sentiment':<12} {'Time'}")
        print(f"  {'─'*62}")

        present = rejected = 0
        for idx, r in enumerate(records, 1):
            icon = "✅" if r[12] == "PRESENT" else "❌"
            sent = r[10] if r[10] else "N/A"
            conf = f"{r[5]:.1f}%"
            ts   = r[14][11:19] if r[14] else "N/A"
            fb   = "💬" if r[8] else "  "
            print(f"  {idx:<4} {r[1]:<14} "
                  f"{icon} {r[12]:<8} "
                  f"{conf:<8} {sent:<12} {ts} {fb}")
            if r[12] == "PRESENT":
                present += 1
            else:
                rejected += 1

        total = len(records)
        rate  = (present/total*100) if total > 0 else 0
        print(f"  {'─'*62}")
        print(f"  Total: {total} | "
              f"Present: {present} ({rate:.1f}%) | "
              f"Rejected: {rejected}")
        print(f"  {'═'*62}")
        return records

    def print_sentiment_report(self, course_code,
                                 date=None):
        """
        D. Sentiment Dashboard.
        Total feedback count, pos/neg/neutral.
        """
        if not self._require_ownership(course_code):
            return

        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        query  = """
            SELECT sentiment, COUNT(*) as cnt,
                   AVG(sentiment_conf) as avg_c
            FROM attendance
            WHERE course_code = ?
              AND sentiment IS NOT NULL
              AND feedback_given = 1
        """
        params = [course_code.upper()]
        if date:
            query  += " AND timestamp LIKE ?"
            params.append(f"{date}%")
        query += " GROUP BY sentiment"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        cursor.execute(
            "SELECT COUNT(*) FROM attendance "
            "WHERE course_code = ? "
            "AND feedback_given = 1"
            + (" AND timestamp LIKE ?" if date else ""),
            [course_code.upper()] +
            ([f"{date}%"] if date else [])
        )
        total = cursor.fetchone()[0]
        conn.close()

        summary = {
            "Positive": {"count": 0, "avg": 0},
            "Negative": {"count": 0, "avg": 0},
            "Neutral" : {"count": 0, "avg": 0}
        }
        for r in rows:
            if r[0] in summary:
                summary[r[0]] = {
                    "count": r[1],
                    "avg"  : round(r[2] or 0, 2)
                }

        today = date or datetime.now().strftime("%Y-%m-%d")
        print(f"\n  {'═'*55}")
        print(f"  SENTIMENT — {course_code} | {today}")
        print(f"  Total feedback: {total}")
        print(f"  {'─'*55}")

        emojis = {"Positive": "😊",
                  "Neutral" : "😐",
                  "Negative": "😟"}
        for label in ["Positive", "Neutral", "Negative"]:
            d   = summary[label]
            pct = (d["count"]/total*100) \
                   if total > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"\n  {emojis[label]} "
                  f"{label:<10}: "
                  f"{d['count']} ({pct:.1f}%)  {bar}")
            print(f"     Avg Confidence: {d['avg']}%")

        if total > 0:
            top = max(summary,
                      key=lambda x: summary[x]["count"])
            print(f"\n  💡 Insight: Students felt "
                  f"mostly {top.lower()}")
        print(f"  {'═'*55}")

    # ─────────────────────────────────────────
    # E. MONITORING & VALIDATION
    # ─────────────────────────────────────────
    def print_monitoring_report(self, course_code):
        """
        E. Security monitoring.
        Detects: spoofed audio, wrong location,
        low confidence rejections.
        """
        if not self._require_ownership(course_code):
            return

        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM attendance
            WHERE course_code = ?
              AND (spoof_verdict    = 'SPOOFED'
               OR  location_verdict = 'BLOCKED'
               OR  status           = 'REJECTED')
            ORDER BY timestamp DESC
        """, (course_code.upper(),))
        records = cursor.fetchall()
        conn.close()

        print(f"\n  {'═'*55}")
        print(f"  🔍 MONITORING — {course_code}")
        print(f"  {'═'*55}")

        if not records:
            print(f"  ✅ No suspicious entries")
            print(f"  {'═'*55}")
            return

        spoofed = sum(1 for r in records
                      if r[6] == "SPOOFED")
        blocked = sum(1 for r in records
                      if r[7] == "BLOCKED")

        print(f"  Total suspicious : {len(records)}")
        print(f"  🔴 Spoofed audio  : {spoofed}")
        print(f"  🔴 Wrong location : {blocked}")
        print(f"  {'─'*55}")

        for r in records:
            if r[6] == "SPOOFED":
                reason = "Spoofed audio"
            elif r[7] == "BLOCKED":
                reason = "Wrong location"
            else:
                reason = f"Low conf ({r[5]:.1f}%)"
            ts = r[14][11:19] if r[14] else "N/A"
            print(f"  ⚠️  {r[1]:<16} "
                  f"{reason:<22} {ts}")
        print(f"  {'═'*55}")

    # ─────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────
    def generate_charts(self, course_code,
                          date=None):
        """Generate attendance + sentiment charts."""
        if not self._require_ownership(course_code):
            return

        today   = date or datetime.now().strftime(
            "%Y-%m-%d")
        records = self.db.get_attendance(
            course_code=course_code, date=date)

        present  = sum(1 for r in records
                       if r[12] == "PRESENT")
        rejected = len(records) - present
        total    = len(records)
        rate     = (present/total*100) if total > 0 else 0

        conn   = sqlite3.connect(self.db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sentiment, COUNT(*)
            FROM attendance
            WHERE course_code = ?
              AND sentiment IS NOT NULL
              AND feedback_given = 1
            GROUP BY sentiment
        """, (course_code.upper(),))
        sent_data = dict(cursor.fetchall())
        conn.close()

        labels = ["Positive", "Neutral", "Negative"]
        counts = [sent_data.get(l, 0) for l in labels]
        colors = ["#4CAF50", "#FF9800", "#F44336"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Attendance bar
        ax = axes[0]
        ax.bar(["Present", "Rejected"],
               [present, rejected],
               color=["#4CAF50", "#F44336"],
               alpha=0.85, width=0.5)
        ax.set_title(
            f"Attendance — {course_code}\n"
            f"Rate: {rate:.1f}% | Total: {total}",
            fontweight="bold")
        ax.set_ylabel("Students")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate([present, rejected]):
            ax.text(i, v + 0.05, str(v),
                    ha="center", fontweight="bold")

        # Attendance pie
        ax2 = axes[1]
        if total > 0:
            ax2.pie(
                [present, rejected],
                labels=["Present", "Rejected"],
                colors=["#4CAF50", "#F44336"],
                autopct="%1.1f%%", startangle=90)
        else:
            ax2.text(0.5, 0.5, "No data",
                     ha="center", va="center",
                     transform=ax2.transAxes)
        ax2.set_title("Attendance Distribution",
                      fontweight="bold")

        # Sentiment bar
        ax3 = axes[2]
        ax3.bar(labels, counts,
                color=colors, alpha=0.85, width=0.5)
        ax3.set_title(
            f"Sentiment — {course_code}\n"
            f"Feedback: {sum(counts)}",
            fontweight="bold")
        ax3.set_ylabel("Responses")
        ax3.grid(axis="y", alpha=0.3)
        for i, v in enumerate(counts):
            if v > 0:
                ax3.text(i, v + 0.05, str(v),
                         ha="center",
                         fontweight="bold")

        plt.suptitle(
            f"Admin Dashboard — "
            f"{course_code} | {today}",
            fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(
            self.PLOTS_DIR,
            f"admin_{course_code}_{today}.png")
        plt.savefig(path, dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"\n  ✅ Chart saved: {path}")
        return path

    def export_report(self, course_code, date=None):
        """Export full attendance report to JSON."""
        if not self._require_ownership(course_code):
            return None

        today   = date or datetime.now().strftime(
            "%Y-%m-%d")
        records = self.db.get_attendance(
            course_code=course_code, date=date)
        present = sum(1 for r in records
                      if r[12] == "PRESENT")

        data = {
            "exported_at": datetime.now().isoformat(),
            "course_code": course_code,
            "lecturer"   : self.current_lecturer["name"],
            "date"       : today,
            "total"      : len(records),
            "present"    : present,
            "rejected"   : len(records) - present,
            "records"    : [
                {
                    "student_id"    : r[1],
                    "status"        : r[12],
                    "confidence"    : r[5],
                    "spoof_verdict" : r[6],
                    "location"      : r[7],
                    "feedback_given": bool(r[8]),
                    "sentiment"     : r[10],
                    "timestamp"     : r[14]
                }
                for r in records
            ]
        }

        os.makedirs("outputs/reports", exist_ok=True)
        path = (f"outputs/reports/"
                f"admin_{course_code}_{today}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  ✅ Report exported: {path}")
        return path

    def show_dashboard(self, course_code):
        """Show full admin dashboard for a course."""
        if not self._require_ownership(course_code):
            return

        today = datetime.now().strftime("%Y-%m-%d")
        self.display_course_page(course_code)
        self.print_attendance_report(
            course_code, today)
        self.print_sentiment_report(
            course_code, today)
        self.print_monitoring_report(course_code)
        self.generate_charts(course_code, today)
        self.export_report(course_code, today)
        print("\n  🎉 Dashboard complete!")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   MODULE 11: ADMINISTRATIVE MODULE TEST")
    print("=" * 60)

    admin = AdminModule()

    # ── A. Authentication
    print("\n[A] Testing Authentication...")

    fail = admin.login("peter@demo.com",
                        "wrongpass", test_mode=True)
    assert not fail
    print("  ✅ Wrong password rejected")

    ok = admin.login("peter@demo.com",
                      "temp123", test_mode=True)
    assert ok

    # Student blocked from lecturer module
    student_user = admin.db.login(
        "austin@student.com", "123456")
    assert student_user["role"] == "student"
    admin.current_lecturer = None
    blocked = (student_user["role"] != "lecturer")
    assert blocked
    print("  ✅ Student blocked from lecturer module")

    admin.login("peter@demo.com",
                 "temp123", test_mode=True)

    # ── B. Course Management
    print("\n[B] Testing Course Management...")

    admin.create_course(
        "CSC400", "Machine Learning",
        "CS Lab Block A, ABUAD",
        "08:00", "10:00"
    )
    admin.edit_course(
        "CSC400",
        venue      = "CS Lab Block B, ABUAD",
        start_time = "09:00",
        end_time   = "11:00"
    )
    admin.view_my_courses()

    course = admin.select_course(
        test_mode=True, test_course="CSC308")
    assert course is not None
    print(f"  ✅ Selected: {course['course_code']}")

    # ── Security: ownership restriction
    print("\n[Security] Testing ownership...")
    admin.login("anna@demo.com",
                 "temp123", test_mode=True)
    result = admin.edit_course(
        "CSC308", venue="Hacked Venue")
    assert not result
    print("  ✅ Cross-course access blocked!")

    admin.login("peter@demo.com",
                 "temp123", test_mode=True)

    # ── C. System Control
    print("\n[C] Testing System Control...")

    admin.start_session("CSC308", test_mode=True)

    # Manual open BOTH
    admin.manual_open_attendance("CSC308")
    admin.manual_open_feedback("CSC308")

    att = admin.can_mark_attendance("CSC308")
    fb  = admin.can_give_feedback("CSC308")
    print(f"  ✅ Can mark attendance : {att}")
    print(f"  ✅ Can give feedback   : {fb}")
    assert att, "Attendance should be OPEN"
    assert fb,  "Feedback should be OPEN"

    # Close attendance only — feedback stays open
    admin.manual_close_attendance("CSC308")
    att2 = admin.can_mark_attendance("CSC308")
    fb2  = admin.can_give_feedback("CSC308")
    print(f"  ✅ After close attendance only:")
    print(f"     Attendance : {att2}")
    print(f"     Feedback   : {fb2}")
    assert not att2, "Attendance should be CLOSED"
    assert fb2,      "Feedback should still be OPEN"

    admin.manual_open_attendance("CSC308")

    # ── Enrollment
    print("\n[Enrollment] Testing...")
    admin.enroll_student("STU001", "CSC308")
    admin.enroll_student("STU002", "CSC308")

    students = admin.db.get_enrolled_students("CSC308")
    print(f"  ✅ Enrolled: {len(students)} students")

    # ── D. Dashboard
    print("\n[D] Testing Dashboard...")
    admin.show_dashboard("CSC308")

    # ── F. Session End
    print("\n[F] Testing Session End...")
    admin.end_session("CSC308")
    assert not admin.can_mark_attendance("CSC308")
    assert not admin.can_give_feedback("CSC308")
    print("  ✅ Session ended — both CLOSED")

    admin.logout()

    print("\n" + "=" * 60)
    print("✅ Administrative Module complete!")
    print("=" * 60)