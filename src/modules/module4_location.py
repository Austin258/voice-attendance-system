import os
import sys
import math
import sqlite3
import json
from datetime import datetime

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────
# MODULE 4: LOCATION VERIFICATION MODULE
# ─────────────────────────────────────────────
class LocationVerificationModule:
    """
    Ensures attendance is marked only within the
    designated classroom.

    Addresses Impersonation/Proxy Attendance by:
    - Verifying exact GPS coordinates
    - Blocking access from outside the classroom
    - Logging all location checks

    Venue: Architecture Department, ABUAD, Ado Ekiti
    Coordinates: 7.6041241, 5.3059950
    Tolerance: 15m (GPS natural drift)
    """

    VENUE = {
        "name"       : "Architecture Department, "
                        "Afe Babalola University, Ado Ekiti",
        "latitude"   : 7.6041241,
        "longitude"  : 5.3059950,
        "tolerance_m": 15
    }

    DB_PATH    = "database/attendance_system.db"
    OUTPUT_DIR = "outputs/features/location"

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._setup_table()

    def _setup_table(self):
        conn   = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  TEXT,
                lat         REAL,
                lon         REAL,
                distance_m  REAL,
                verdict     TEXT,
                allowed     INTEGER,
                timestamp   TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────
    # HAVERSINE FORMULA
    # ─────────────────────────────────────────
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance in meters between two GPS points."""
        R    = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a    = (math.sin(dphi/2)**2 +
                math.cos(phi1) * math.cos(phi2) *
                math.sin(dlam/2)**2)
        return round(R * 2 * math.atan2(
            math.sqrt(a), math.sqrt(1-a)), 2)

    # ─────────────────────────────────────────
    # REQUEST PERMISSION (terminal simulation)
    # ─────────────────────────────────────────
    def request_permission(self, test_mode=False,
                            test_grant=True):
        """
        Request location permission from device.
        In web deployment: browser Geolocation API.
        In terminal: simulated prompt.
        """
        print("\n  📍 Location Permission Required")
        print(f"     Venue: {self.VENUE['name']}")
        print(f"     This system needs your GPS location")
        print(f"     to verify classroom presence.")

        if test_mode:
            print(f"  ℹ️  Test mode: permission "
                  f"{'GRANTED' if test_grant else 'DENIED'}")
            return test_grant

        ans = input("\n  Allow location access? (yes/no): "
                    ).strip().lower()
        return ans == "yes"

    # ─────────────────────────────────────────
    # VERIFY LOCATION
    # ─────────────────────────────────────────
    def verify(self, lat, lon, student_id="unknown"):
        """
        Verify if student is at exact classroom location.
        Returns result dict with verdict and distance.
        """
        distance = self._haversine(
            lat, lon,
            self.VENUE["latitude"],
            self.VENUE["longitude"]
        )
        allowed = distance <= self.VENUE["tolerance_m"]
        verdict = "ALLOWED" if allowed else "BLOCKED"

        if allowed:
            message = (
                f"✅ Exact location verified. "
                f"You are {distance}m from "
                f"{self.VENUE['name']}. You may proceed."
            )
        else:
            message = (
                f"❌ Access denied. You are {distance}m "
                f"away from the classroom. "
                f"Please be physically present."
            )

        result = {
            "student_id"  : student_id,
            "lat"         : lat,
            "lon"         : lon,
            "venue_name"  : self.VENUE["name"],
            "distance_m"  : distance,
            "tolerance_m" : self.VENUE["tolerance_m"],
            "verdict"     : verdict,
            "allowed"     : allowed,
            "message"     : message,
            "timestamp"   : datetime.now().isoformat()
        }

        self._log(result)
        return result

    # ─────────────────────────────────────────
    # LOG TO DATABASE
    # ─────────────────────────────────────────
    def _log(self, result):
        try:
            conn   = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO location_log
                (student_id, lat, lon, distance_m,
                 verdict, allowed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result["student_id"],
                result["lat"], result["lon"],
                result["distance_m"],
                result["verdict"],
                int(result["allowed"]),
                result["timestamp"]
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ─────────────────────────────────────────
    # UPDATE VENUE (admin function)
    # ─────────────────────────────────────────
    def update_venue(self, name, lat, lon, tolerance=15):
        """
        Update classroom venue coordinates.
        Called by admin/lecturer before class.
        """
        self.VENUE["name"]        = name
        self.VENUE["latitude"]    = lat
        self.VENUE["longitude"]   = lon
        self.VENUE["tolerance_m"] = tolerance
        print(f"  ✅ Venue updated: {name}")
        print(f"     Coordinates : ({lat}, {lon})")
        print(f"     Tolerance   : {tolerance}m")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   MODULE 4: LOCATION VERIFICATION TEST")
    print("=" * 55)

    loc = LocationVerificationModule()

    tests = [
        ("AT venue",     7.6041241, 5.3059950),
        ("NEAR venue",   7.6042000, 5.3060500),
        ("OFF campus",   7.6200000, 5.3300000),
        ("Abuja",        9.0579000, 7.4951000),
    ]

    for label, lat, lon in tests:
        r = loc.verify(lat, lon, "test_student")
        print(f"\n  📌 {label}")
        print(f"     {r['message']}")

    print("\n✅ Location Verification Module working!")