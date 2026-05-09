import os
import sys
import sqlite3
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# REPORTING AND DASHBOARD MODULE
# ─────────────────────────────────────────────
class DashboardModule:
    """
    Reporting and dashboard for attendance and sentiment.

    Features:
    - Attendance list per course per date
    - Sentiment summary and breakdown
    - Attendance rate charts
    - Sentiment distribution charts
    - Per-student attendance history
    - Export attendance to JSON/CSV
    """

    DB_PATH   = "database/attendance_system.db"
    PLOTS_DIR = "outputs/plots/dashboard"

    def __init__(self):
        os.makedirs(self.PLOTS_DIR, exist_ok=True)

    # ─────────────────────────────────────────
    # ATTENDANCE LIST
    # ─────────────────────────────────────────
    def get_attendance_list(self, course_code=None,
                             date=None, status=None):
        """
        Get formatted attendance list.
        Filters by course, date, and status.
        """
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        query  = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if course_code:
            query += " AND course_code = ?"
            params.append(course_code)
        if date:
            query += " AND timestamp LIKE ?"
            params.append(f"{date}%")
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC"
        cursor.execute(query, params)
        records = cursor.fetchall()
        conn.close()
        return records

    # ─────────────────────────────────────────
    # PRINT ATTENDANCE TABLE
    # ─────────────────────────────────────────
    def print_attendance_table(self, course_code=None,
                                date=None):
        """Print a formatted attendance table."""
        records = self.get_attendance_list(
            course_code=course_code, date=date)

        title = "ATTENDANCE LIST"
        if course_code:
            title += f" — {course_code}"
        if date:
            title += f" — {date}"

        print(f"\n  {'═'*60}")
        print(f"  {title:^60}")
        print(f"  {'═'*60}")
        print(f"  {'#':<4} {'Student':<14} {'Status':<10} "
              f"{'Confidence':<12} {'Sentiment':<12} "
              f"{'Time'}")
        print(f"  {'─'*4}─{'─'*14}─{'─'*10}─{'─'*12}─"
              f"{'─'*12}─{'─'*8}")

        present  = 0
        rejected = 0

        for idx, r in enumerate(records, 1):
            status    = r[9]
            sentiment = r[7] if r[7] else "N/A"
            conf      = f"{r[2]:.1f}%"
            time_str  = r[13][11:19] if r[13] else "N/A"
            s_icon    = "✅" if status == "PRESENT" else "❌"

            print(f"  {idx:<4} {r[1]:<14} "
                  f"{s_icon} {status:<8} "
                  f"{conf:<12} {sentiment:<12} {time_str}")

            if status == "PRESENT":
                present += 1
            else:
                rejected += 1

        total = len(records)
        rate  = (present/total*100) if total > 0 else 0

        print(f"  {'─'*60}")
        print(f"  Total     : {total}")
        print(f"  Present   : {present} ({rate:.1f}%)")
        print(f"  Rejected  : {rejected}")
        print(f"  {'═'*60}")

        return records

    # ─────────────────────────────────────────
    # SENTIMENT SUMMARY
    # ─────────────────────────────────────────
    def get_sentiment_summary(self, course_code=None,
                               date=None):
        """Get sentiment breakdown for a course/date."""
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        query  = """
            SELECT sentiment, COUNT(*) as count,
                   AVG(sentiment_conf) as avg_conf
            FROM attendance
            WHERE sentiment IS NOT NULL
              AND feedback_given = 1
        """
        params = []

        if course_code:
            query += " AND course_code = ?"
            params.append(course_code)
        if date:
            query += " AND timestamp LIKE ?"
            params.append(f"{date}%")

        query += " GROUP BY sentiment"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Total with feedback
        cursor.execute("""
            SELECT COUNT(*) FROM attendance
            WHERE feedback_given = 1
        """ + (" AND course_code = ?" if course_code else ""),
        ([course_code] if course_code else []))
        total_feedback = cursor.fetchone()[0]

        conn.close()

        summary = {
            "Positive": {"count": 0, "avg_conf": 0},
            "Negative": {"count": 0, "avg_conf": 0},
            "Neutral" : {"count": 0, "avg_conf": 0},
        }
        for row in rows:
            if row[0] in summary:
                summary[row[0]] = {
                    "count"   : row[1],
                    "avg_conf": round(row[2] or 0, 2)
                }

        return summary, total_feedback

    # ─────────────────────────────────────────
    # PRINT SENTIMENT SUMMARY
    # ─────────────────────────────────────────
    def print_sentiment_summary(self, course_code=None,
                                  date=None):
        """Print formatted sentiment summary."""
        summary, total = self.get_sentiment_summary(
            course_code=course_code, date=date)

        title = "SENTIMENT SUMMARY"
        if course_code:
            title += f" — {course_code}"

        print(f"\n  {'═'*55}")
        print(f"  {title:^55}")
        print(f"  {'═'*55}")
        print(f"  Total feedback responses : {total}")
        print(f"  {'─'*55}")

        emojis = {
            "Positive": "😊",
            "Neutral" : "😐",
            "Negative": "😟"
        }

        for label in ["Positive", "Neutral", "Negative"]:
            data  = summary[label]
            count = data["count"]
            pct   = (count/total*100) if total > 0 else 0
            bar   = "█" * int(pct / 5)
            emoji = emojis[label]

            print(f"\n  {emoji} {label:<10}: {count:>3} responses "
                  f"({pct:.1f}%)")
            print(f"     Bar      : {bar}")
            print(f"     Avg Conf : {data['avg_conf']}%")

        print(f"\n  {'═'*55}")

        # Insight
        max_label = max(summary,
                        key=lambda x: summary[x]["count"])
        if total > 0:
            print(f"  💡 Insight: Most students felt "
                  f"{max_label.lower()} about the lecture")
        print(f"  {'═'*55}")

        return summary

    # ─────────────────────────────────────────
    # PLOT ATTENDANCE CHART
    # ─────────────────────────────────────────
    def plot_attendance_chart(self, course_code=None):
        """Bar chart showing present vs rejected."""
        records = self.get_attendance_list(
            course_code=course_code)

        present  = sum(1 for r in records
                       if r[9] == "PRESENT")
        rejected = sum(1 for r in records
                       if r[9] == "REJECTED")
        total    = len(records)
        rate     = (present/total*100) if total > 0 else 0

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(["Present", "Rejected"],
                      [present, rejected],
                      color=["#4CAF50", "#F44336"],
                      alpha=0.85, edgecolor="white",
                      width=0.5)
        ax.set_title(
            f"Attendance Status\n"
            f"{course_code or 'All Courses'} — "
            f"Rate: {rate:.1f}%",
            fontweight="bold", fontsize=12)
        ax.set_ylabel("Number of Students")
        ax.set_ylim(0, max(present, rejected) * 1.3 + 1)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h}",
                        xy=(bar.get_x() +
                            bar.get_width()/2, h),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", fontweight="bold",
                        fontsize=12)

        # Pie chart
        ax2 = axes[1]
        if total > 0:
            ax2.pie([present, rejected],
                    labels=["Present", "Rejected"],
                    colors=["#4CAF50", "#F44336"],
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 11})
            ax2.set_title("Attendance Distribution",
                          fontweight="bold", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "No data",
                     ha="center", va="center",
                     transform=ax2.transAxes)

        plt.tight_layout()
        path = os.path.join(
            self.PLOTS_DIR,
            f"attendance_{course_code or 'all'}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  ✅ Attendance chart saved: {path}")
        return path

    # ─────────────────────────────────────────
    # PLOT SENTIMENT CHART
    # ─────────────────────────────────────────
    def plot_sentiment_chart(self, course_code=None):
        """Pie and bar chart for sentiment breakdown."""
        summary, total = self.get_sentiment_summary(
            course_code=course_code)

        labels  = ["Positive", "Neutral", "Negative"]
        counts  = [summary[l]["count"] for l in labels]
        colors  = ["#4CAF50", "#FF9800", "#F44336"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        ax = axes[0]
        if sum(counts) > 0:
            ax.pie(counts, labels=labels, colors=colors,
                   autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 11})
        else:
            ax.text(0.5, 0.5, "No feedback data",
                    ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(
            f"Sentiment Distribution\n"
            f"{course_code or 'All Courses'}",
            fontweight="bold", fontsize=12)

        # Bar chart with confidence
        ax2    = axes[1]
        confs  = [summary[l]["avg_conf"] for l in labels]
        x      = np.arange(len(labels))
        bars1  = ax2.bar(x - 0.2, counts, 0.35,
                         label="Count",
                         color=colors, alpha=0.85)
        ax2.set_title("Sentiment Count & Confidence",
                      fontweight="bold", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Count")
        ax2.grid(axis="y", alpha=0.3)

        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax2.annotate(f"{h}",
                             xy=(bar.get_x() +
                                 bar.get_width()/2, h),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha="center", fontsize=10)

        plt.tight_layout()
        path = os.path.join(
            self.PLOTS_DIR,
            f"sentiment_{course_code or 'all'}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Sentiment chart saved: {path}")
        return path

    # ─────────────────────────────────────────
    # STUDENT ATTENDANCE HISTORY
    # ─────────────────────────────────────────
    def get_student_history(self, student_id):
        """Get full attendance history for a student."""
        records = self.get_attendance_list(
            course_code=None)
        student_records = [
            r for r in records
            if r[1] == student_id
        ]

        print(f"\n  📋 Attendance History: {student_id}")
        print(f"  {'─'*50}")

        if not student_records:
            print(f"  No records found for {student_id}")
            return []

        for r in student_records:
            status = "✅" if r[9] == "PRESENT" else "❌"
            sent   = r[7] if r[7] else "N/A"
            ts     = r[13][:19] if r[13] else "N/A"
            course = r[10] if r[10] else "N/A"
            print(f"  {status} {ts} | "
                  f"{course:<10} | {sent}")

        present = sum(1 for r in student_records
                      if r[9] == "PRESENT")
        total   = len(student_records)
        rate    = (present/total*100) if total > 0 else 0
        print(f"\n  Attendance Rate: {present}/{total} "
              f"({rate:.1f}%)")

        return student_records

    # ─────────────────────────────────────────
    # EXPORT TO JSON
    # ─────────────────────────────────────────
    def export_attendance_json(self, course_code=None,
                                date=None):
        """Export attendance records to JSON file."""
        records = self.get_attendance_list(
            course_code=course_code, date=date)

        data = []
        for r in records:
            data.append({
                "id"              : r[0],
                "student_id"      : r[1],
                "confidence"      : r[2],
                "spoof_verdict"   : r[3],
                "location_verdict": r[4],
                "feedback_given"  : bool(r[5]),
                "feedback_text"   : r[6],
                "sentiment"       : r[7],
                "sentiment_conf"  : r[8],
                "status"          : r[9],
                "course_code"     : r[10],
                "course_title"    : r[11],
                "timestamp"       : r[13]
            })

        filename = (
            f"attendance_"
            f"{course_code or 'all'}_"
            f"{date or 'all'}.json"
        )
        path = os.path.join(
            "outputs/reports", filename)
        os.makedirs("outputs/reports", exist_ok=True)

        with open(path, "w") as f:
            json.dump({
                "exported_at" : datetime.now().isoformat(),
                "course"      : course_code,
                "date"        : date,
                "total"       : len(data),
                "records"     : data
            }, f, indent=2)

        print(f"\n  ✅ Exported {len(data)} records to: {path}")
        return path

    # ─────────────────────────────────────────
    # FULL DASHBOARD DISPLAY
    # ─────────────────────────────────────────
    def show_dashboard(self, course_code=None):
        """Show complete dashboard summary."""
        today = datetime.now().strftime("%Y-%m-%d")

        print("\n" + "=" * 60)
        print("  📊 ATTENDANCE & SENTIMENT DASHBOARD")
        if course_code:
            print(f"  Course: {course_code}")
        print(f"  Date  : {today}")
        print("=" * 60)

        # Attendance table
        self.print_attendance_table(
            course_code=course_code, date=today)

        # Sentiment summary
        self.print_sentiment_summary(
            course_code=course_code, date=today)

        # Generate charts
        print("\n  📈 Generating charts...")
        self.plot_attendance_chart(course_code)
        self.plot_sentiment_chart(course_code)

        # Export
        self.export_attendance_json(
            course_code=course_code, date=today)

        print("\n  🎉 Dashboard complete!")
        print("=" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DASHBOARD MODULE TEST")
    print("=" * 55)

    dash = DashboardModule()
    dash.show_dashboard()

    # Test student history
    dash.get_student_history("student14")

    print("\n✅ Dashboard Module working!")