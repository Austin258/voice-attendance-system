import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# MODULE 1: USER INTERFACE MODULE
# ─────────────────────────────────────────────
class UIModule:
    """
    Provides the interaction layer between users and system.
    Handles registration, login, prompts, and result display.
    In deployment this connects to the web frontend.
    In terminal mode it uses structured text prompts.
    """

    def __init__(self):
        self.system_name = (
            "AI Voice-Activated Attendance & Sentiment System"
        )
        self.institution = (
            "Architecture Department, Afe Babalola University"
        )

    # ─────────────────────────────────────────
    # DISPLAY WELCOME BANNER
    # ─────────────────────────────────────────
    def show_banner(self):
        print("\n" + "=" * 55)
        print(f"  {self.system_name}")
        print(f"  {self.institution}")
        print(f"  {datetime.now().strftime('%A, %d %B %Y — %H:%M')}")
        print("=" * 55)

    # ─────────────────────────────────────────
    # MAIN MENU
    # ─────────────────────────────────────────
    def show_main_menu(self, test_mode=False,
                        test_choice=None):
        """
        Display main menu and get user selection.
        Options:
          1. Mark Attendance
          2. View My Attendance
          3. Exit
        """
        print("\n  📋 MAIN MENU")
        print("  ─────────────────────────────")
        print("  [1] Mark Attendance")
        print("  [2] View My Attendance Record")
        print("  [3] Exit")
        print("  ─────────────────────────────")

        if test_mode and test_choice is not None:
            print(f"  ℹ️  Test mode choice: {test_choice}")
            return str(test_choice)

        choice = input("  Enter choice (1/2/3): ").strip()
        return choice

    # ─────────────────────────────────────────
    # ATTENDANCE PROMPT
    # ─────────────────────────────────────────
    def show_attendance_prompt(self):
        """Display instructions before voice recording."""
        print("\n" + "─" * 55)
        print("  🎤 ATTENDANCE MARKING")
        print("─" * 55)
        print("\n  Please follow these steps:")
        print("  1️⃣  Ensure you are in the classroom")
        print("  2️⃣  Hold your device at normal speaking distance")
        print("  3️⃣  When prompted, say the phrase clearly:")
        print('      "My name is [Your Name], I am present"')
        print("  4️⃣  Speak clearly and avoid background noise")
        print("\n  ⚠️  Your location will be verified automatically")
        print("  ⚠️  Live voice detection is active")

    # ─────────────────────────────────────────
    # FEEDBACK PROMPT
    # ─────────────────────────────────────────
    def show_feedback_prompt(self, test_mode=False,
                              test_choice=None):
        """
        Ask student if they want to provide feedback.
        Returns True (yes) or False (skip).
        """
        print("\n" + "─" * 55)
        print("  💬 FEEDBACK (OPTIONAL)")
        print("─" * 55)
        print("\n  Your attendance has been marked ✅")
        print("\n  Would you like to give feedback on")
        print("  today's lecture?")
        print("\n  [Y] Yes — Record feedback (5–30 seconds)")
        print("  [N] Skip — No feedback today")

        if test_mode and test_choice is not None:
            print(f"\n  ℹ️  Test mode: {test_choice}")
            return str(test_choice).upper() in ["Y", "YES"]

        choice = input("\n  Your choice (Y/N): ").strip().upper()
        return choice in ["Y", "YES"]

    # ─────────────────────────────────────────
    # DISPLAY ATTENDANCE RESULT
    # ─────────────────────────────────────────
    def show_attendance_result(self, result):
        """Display formatted attendance result to user."""
        print("\n" + "=" * 55)
        print("  📊 ATTENDANCE RESULT")
        print("=" * 55)

        status_icon = (
            "✅ PRESENT" if result.get("status") == "PRESENT"
            else "❌ REJECTED"
        )

        print(f"\n  Status          : {status_icon}")
        print(f"  Student         : {result.get('student_id', 'N/A')}")
        print(f"  Confidence      : {result.get('confidence', 0)}%")
        print(f"  Location        : {result.get('location', 'N/A')}")
        print(f"  Spoof Check     : {result.get('spoof_check', 'N/A')}")
        print(f"  Timestamp       : {result.get('timestamp', 'N/A')}")

        if result.get("feedback_given"):
            print(f"\n  💬 FEEDBACK RESULT")
            print(f"  Sentiment       : "
                  f"{result.get('sentiment', 'N/A')} "
                  f"({result.get('sentiment_conf', 0)}%)")
        else:
            print(f"\n  💬 Feedback     : Not Provided")

        if result.get("reason"):
            print(f"\n  ℹ️  Note: {result.get('reason')}")

        print("=" * 55)

    # ─────────────────────────────────────────
    # DISPLAY ATTENDANCE RECORDS
    # ─────────────────────────────────────────
    def show_attendance_records(self, records):
        """Display a student's attendance history."""
        print("\n" + "=" * 55)
        print("  📋 ATTENDANCE RECORDS")
        print("=" * 55)

        if not records:
            print("\n  No attendance records found.")
            return

        print(f"\n  {'#':<4} {'Student':<14} {'Status':<10} "
              f"{'Sentiment':<12} {'Date'}")
        print(f"  {'─'*4}─{'─'*14}─{'─'*10}─"
              f"{'─'*12}─{'─'*19}")

        for idx, r in enumerate(records, 1):
            sentiment = r[6] if r[6] else "N/A"
            timestamp = r[9][:19] if r[9] else "N/A"
            print(f"  {idx:<4} {r[1]:<14} {r[8]:<10} "
                  f"{sentiment:<12} {timestamp}")

        print(f"\n  Total records: {len(records)}")
        print("=" * 55)

    # ─────────────────────────────────────────
    # DISPLAY ERROR
    # ─────────────────────────────────────────
    def show_error(self, message):
        print(f"\n  ❌ ERROR: {message}")

    # ─────────────────────────────────────────
    # DISPLAY SUCCESS
    # ─────────────────────────────────────────
    def show_success(self, message):
        print(f"\n  ✅ {message}")

    # ─────────────────────────────────────────
    # DISPLAY INFO
    # ─────────────────────────────────────────
    def show_info(self, message):
        print(f"\n  ℹ️  {message}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 1: USER INTERFACE TEST")
    print("=" * 55)

    ui = UIModule()
    ui.show_banner()
    ui.show_attendance_prompt()
    ui.show_feedback_prompt(test_mode=True, test_choice="Y")

    sample_result = {
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
    }
    ui.show_attendance_result(sample_result)
    print("\n✅ UI Module working correctly!")