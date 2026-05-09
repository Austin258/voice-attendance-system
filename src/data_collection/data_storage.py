import os
import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DB_PATH       = "database/attendance_system.db"
VOICE_DIR     = "data/voice_samples"
SENTIMENT_CSV = "data/sentiment_feedback/transcripts/sentiment_dataset.csv"
N_STUDENTS    = 32
N_RECORDINGS  = 7

os.makedirs("database", exist_ok=True)

# ─────────────────────────────────────────────
# 1. DATABASE SETUP — Create Tables
# ─────────────────────────────────────────────
def create_tables(conn):
    cursor = conn.cursor()

    # Students table (anonymized — no real names)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id      TEXT PRIMARY KEY,
            hashed_id       TEXT NOT NULL,
            gender          TEXT DEFAULT 'Male',
            registered_at   TEXT NOT NULL
        )
    """)

    # Voice samples metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_samples (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT NOT NULL,
            filename        TEXT NOT NULL,
            file_path       TEXT NOT NULL,
            sample_rate     INTEGER DEFAULT 16000,
            format          TEXT DEFAULT 'wav',
            recorded_at     TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    # Sentiment feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_feedback (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id          TEXT NOT NULL,
            question            TEXT NOT NULL,
            response            TEXT NOT NULL,
            compound_score      REAL,
            sentiment_label     TEXT NOT NULL,
            sentiment_encoded   INTEGER,
            collected_at        TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    # Access log table (ethical compliance tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            action      TEXT NOT NULL,
            table_name  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            status      TEXT NOT NULL
        )
    """)

    conn.commit()
    print("✅ All database tables created successfully")

# ─────────────────────────────────────────────
# 2. ANONYMIZATION — Hash student IDs
# ─────────────────────────────────────────────
def anonymize_id(student_id):
    """Hash student ID using SHA-256 for privacy protection."""
    return hashlib.sha256(student_id.encode()).hexdigest()[:16]

# ─────────────────────────────────────────────
# 3. LOG ACCESS EVENTS
# ─────────────────────────────────────────────
def log_action(conn, action, table_name, status="SUCCESS"):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO access_log (action, table_name, timestamp, status)
        VALUES (?, ?, ?, ?)
    """, (action, table_name, datetime.now().isoformat(), status))
    conn.commit()

# ─────────────────────────────────────────────
# 4. POPULATE STUDENTS TABLE
# ─────────────────────────────────────────────
def populate_students(conn):
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    count = 0

    for i in range(1, N_STUDENTS + 1):
        student_id = f"student{i}"
        hashed     = anonymize_id(student_id)
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO students
                (student_id, hashed_id, gender, registered_at)
                VALUES (?, ?, ?, ?)
            """, (student_id, hashed, "Male", now))
            count += 1
        except Exception as e:
            print(f"⚠️  Could not insert {student_id}: {e}")

    conn.commit()
    log_action(conn, "INSERT", "students")
    print(f"✅ {count} students registered in database (anonymized)")

# ─────────────────────────────────────────────
# 5. POPULATE VOICE SAMPLES TABLE
# ─────────────────────────────────────────────
def populate_voice_samples(conn):
    cursor = conn.cursor()
    now    = datetime.now().isoformat()
    count  = 0

    for i in range(1, N_STUDENTS + 1):
        student_id = f"student{i}"
        for j in range(1, N_RECORDINGS + 1):
            filename  = f"{student_id}_{j}.wav"
            file_path = os.path.join(VOICE_DIR, student_id, filename)

            if os.path.exists(file_path):
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO voice_samples
                        (student_id, filename, file_path,
                         sample_rate, format, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (student_id, filename, file_path,
                          16000, "wav", now))
                    count += 1
                except Exception as e:
                    print(f"⚠️  Error inserting {filename}: {e}")

    conn.commit()
    log_action(conn, "INSERT", "voice_samples")
    print(f"✅ {count} voice sample records stored in database")

# ─────────────────────────────────────────────
# 6. POPULATE SENTIMENT FEEDBACK TABLE
# ─────────────────────────────────────────────
def populate_sentiment_feedback(conn):
    cursor = conn.cursor()
    now    = datetime.now().isoformat()

    if not os.path.exists(SENTIMENT_CSV):
        print(f"❌ Sentiment CSV not found: {SENTIMENT_CSV}")
        return

    df    = pd.read_csv(SENTIMENT_CSV)
    count = 0

    label_map = {"Positive": 2, "Negative": 0, "Neutral": 1}

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO sentiment_feedback
                (student_id, question, response, compound_score,
                 sentiment_label, sentiment_encoded, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["student_id"],
                row["question"],
                row["response"],
                row["compound_score"],
                row["sentiment_label"],
                label_map.get(row["sentiment_label"], 1),
                now
            ))
            count += 1
        except Exception as e:
            print(f"⚠️  Error inserting feedback row: {e}")

    conn.commit()
    log_action(conn, "INSERT", "sentiment_feedback")
    print(f"✅ {count} sentiment feedback records stored in database")

# ─────────────────────────────────────────────
# 7. VERIFY DATABASE CONTENTS
# ─────────────────────────────────────────────
def verify_database(conn):
    cursor = conn.cursor()

    tables = ["students", "voice_samples", "sentiment_feedback", "access_log"]

    print("\n" + "=" * 55)
    print("   DATABASE VERIFICATION")
    print("=" * 55)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"   📋 {table:<25}: {count} records")

    # Show sample student records
    print("\n🔍 Sample Students (anonymized):")
    cursor.execute("SELECT student_id, hashed_id, gender FROM students LIMIT 3")
    for row in cursor.fetchall():
        print(f"   ID: {row[0]} | Hash: {row[1]} | Gender: {row[2]}")

    # Show sentiment distribution
    print("\n📊 Sentiment Distribution in DB:")
    cursor.execute("""
        SELECT sentiment_label, COUNT(*) as count
        FROM sentiment_feedback
        GROUP BY sentiment_label
    """)
    for row in cursor.fetchall():
        bar = "█" * row[1]
        print(f"   {row[0]:<10}: {row[1]:>3}  {bar}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DATA STORAGE AND MANAGEMENT")
    print("=" * 55)

    conn = sqlite3.connect(DB_PATH)
    print(f"\n✅ Connected to database: {DB_PATH}")

    create_tables(conn)
    populate_students(conn)
    populate_voice_samples(conn)
    populate_sentiment_feedback(conn)
    verify_database(conn)

    conn.close()

    print("\n" + "=" * 55)
    print("✅ Data storage complete!")
    print(f"✅ Database saved at: {DB_PATH}")
    print("🚀 Ready to proceed to preprocessing")
    print("=" * 55)