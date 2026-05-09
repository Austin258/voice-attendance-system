import os
import sys
import sqlite3
from datetime import datetime

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# SYSTEM CONTROL MODULE
# ─────────────────────────────────────────────
class SystemControlModule:
    """
    Controls when attendance and feedback are open.

    Lecturer/Admin functions:
    - Open attendance for a specific course
    - Close attendance
    - Open feedback collection
    - Close feedback
    - View current status

    Students can only mark attendance or give feedback
    when the lecturer has explicitly opened it.
    """

    DB_PATH = "database/attendance_system.db"

    def __init__(self):
        self._setup_table()

    def _setup_table(self):
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_control (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code          TEXT NOT NULL,
                attendance_open      INTEGER DEFAULT 0,
                feedback_open        INTEGER DEFAULT 0,
                opened_by            TEXT,
                attendance_opened_at TEXT,
                feedback_opened_at   TEXT,
                closed_at            TEXT,
                date                 TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # GET CURRENT STATUS
    # ─────────────────────────────────────────
    def get_status(self, course_code):
        """
        Get current open/closed status for a course.
        Returns dict with attendance_open and feedback_open.
        """
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM system_control
            WHERE course_code = ?
            AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code,
               datetime.now().strftime("%Y-%m-%d")))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                "course_code"      : course_code,
                "attendance_open"  : False,
                "feedback_open"    : False,
                "opened_by"        : None,
                "attendance_opened_at": None,
                "feedback_opened_at"  : None,
                "date"             : datetime.now().strftime(
                    "%Y-%m-%d")
            }

        return {
            "course_code"         : row[1],
            "attendance_open"     : bool(row[2]),
            "feedback_open"       : bool(row[3]),
            "opened_by"           : row[4],
            "attendance_opened_at": row[5],
            "feedback_opened_at"  : row[6],
            "closed_at"           : row[7],
            "date"                : row[8]
        }

    # ─────────────────────────────────────────
    # OPEN ATTENDANCE
    # ─────────────────────────────────────────
    def open_attendance(self, course_code,
                         opened_by="Lecturer"):
        """
        Open attendance for a course.
        Only one session can be open per course per day.
        """
        status = self.get_status(course_code)
        now    = datetime.now().isoformat()
        today  = datetime.now().strftime("%Y-%m-%d")

        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        if status["attendance_open"]:
            print(f"  ⚠️  Attendance already open "
                  f"for {course_code}")
            conn.close()
            return False

        cursor.execute("""
            INSERT INTO system_control
            (course_code, attendance_open, feedback_open,
             opened_by, attendance_opened_at, date)
            VALUES (?, 1, 0, ?, ?, ?)
        """, (course_code, opened_by, now, today))
        conn.commit()
        conn.close()

        print(f"\n  ✅ Attendance OPENED for {course_code}")
        print(f"     By    : {opened_by}")
        print(f"     Time  : {now[:19]}")
        return True

    # ─────────────────────────────────────────
    # CLOSE ATTENDANCE
    # ─────────────────────────────────────────
    def close_attendance(self, course_code):
        """Close attendance for a course."""
        now  = datetime.now().isoformat()
        today = datetime.now().strftime("%Y-%m-%d")

        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE system_control
            SET attendance_open = 0, closed_at = ?
            WHERE course_code = ? AND date = ?
            AND attendance_open = 1
        """, (now, course_code, today))
        updated = cursor.rowcount
        conn.commit()
        conn.close()

        if updated:
            print(f"\n  🔒 Attendance CLOSED for {course_code}")
            return True
        print(f"\n  ⚠️  No open attendance found for "
              f"{course_code}")
        return False

    # ─────────────────────────────────────────
    # OPEN FEEDBACK
    # ─────────────────────────────────────────
    def open_feedback(self, course_code,
                       opened_by="Lecturer"):
        """Open feedback collection for a course."""
        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().isoformat()

        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Check if a session exists for today
        cursor.execute("""
            SELECT id FROM system_control
            WHERE course_code = ? AND date = ?
            ORDER BY id DESC LIMIT 1
        """, (course_code, today))
        row = cursor.fetchone()

        if row:
            cursor.execute("""
                UPDATE system_control
                SET feedback_open = 1,
                    feedback_opened_at = ?
                WHERE id = ?
            """, (now, row[0]))
        else:
            cursor.execute("""
                INSERT INTO system_control
                (course_code, attendance_open, feedback_open,
                 opened_by, feedback_opened_at, date)
                VALUES (?, 0, 1, ?, ?, ?)
            """, (course_code, opened_by, now, today))

        conn.commit()
        conn.close()

        print(f"\n  ✅ Feedback OPENED for {course_code}")
        print(f"     By    : {opened_by}")
        print(f"     Time  : {now[:19]}")
        return True

    # ─────────────────────────────────────────
    # CLOSE FEEDBACK
    # ─────────────────────────────────────────
    def close_feedback(self, course_code):
        """Close feedback for a course."""
        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().isoformat()

        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE system_control
            SET feedback_open = 0, closed_at = ?
            WHERE course_code = ? AND date = ?
            AND feedback_open = 1
        """, (now, course_code, today))
        updated = cursor.rowcount
        conn.commit()
        conn.close()

        if updated:
            print(f"\n  🔒 Feedback CLOSED for {course_code}")
            return True
        print(f"\n  ⚠️  No open feedback found for "
              f"{course_code}")
        return False

    # ─────────────────────────────────────────
    # CHECK ACCESS (student-facing)
    # ─────────────────────────────────────────
    def can_mark_attendance(self, course_code):
        """Check if student can mark attendance now."""
        status = self.get_status(course_code)
        return status["attendance_open"]

    def can_give_feedback(self, course_code):
        """Check if student can give feedback now."""
        status = self.get_status(course_code)
        return status["feedback_open"]

    # ─────────────────────────────────────────
    # DISPLAY STATUS
    # ─────────────────────────────────────────
    def display_status(self, course_code=None):
        """Display current system control status."""
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        today  = datetime.now().strftime("%Y-%m-%d")

        if course_code:
            cursor.execute("""
                SELECT * FROM system_control
                WHERE course_code = ? AND date = ?
                ORDER BY id DESC LIMIT 1
            """, (course_code, today))
            rows = cursor.fetchall()
            rows = rows if rows else []
        else:
            cursor.execute("""
                SELECT * FROM system_control
                WHERE date = ?
                ORDER BY course_code
            """, (today,))
            rows = cursor.fetchall()

        conn.close()

        print("\n  📋 System Control Status")
        print(f"  Date: {today}")
        print("  ─────────────────────────────────────")
        print(f"  {'Course':<10} {'Attendance':<14} "
              f"{'Feedback':<12} {'By'}")
        print(f"  {'─'*10}─{'─'*14}─{'─'*12}─{'─'*10}")

        if not rows:
            print("  No sessions found for today.")
            return

        for r in rows:
            att = "🟢 OPEN" if r[2] else "🔴 CLOSED"
            fb  = "🟢 OPEN" if r[3] else "🔴 CLOSED"
            print(f"  {r[1]:<10} {att:<14} {fb:<12} "
                  f"{r[4] or 'N/A'}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   SYSTEM CONTROL MODULE TEST")
    print("=" * 55)

    ctrl = SystemControlModule()

    print("\n[1] Opening attendance for ARC101...")
    ctrl.open_attendance("ARC101", "Dr. Admin")

    print("\n[2] Opening feedback for ARC101...")
    ctrl.open_feedback("ARC101", "Dr. Admin")

    print("\n[3] Current status:")
    ctrl.display_status()

    print("\n[4] Can student mark attendance?",
          ctrl.can_mark_attendance("ARC101"))
    print("    Can student give feedback?",
          ctrl.can_give_feedback("ARC101"))

    print("\n[5] Closing attendance...")
    ctrl.close_attendance("ARC101")

    print("\n[6] Updated status:")
    ctrl.display_status()

    print("\n✅ System Control Module working!")