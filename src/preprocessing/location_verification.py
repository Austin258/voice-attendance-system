import math
from datetime import datetime

# ─────────────────────────────────────────────
# VENUE DEFINITIONS
# Coordinates computed from front/middle/back
# of each lab using center + radius method
# ─────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
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
        math.sqrt(a), math.sqrt(1 - a)), 2)

def _compute_center_radius(points, buffer=5):
    """
    Compute center lat/lon and radius from
    front/middle/back points.
    Radius = max distance from center + buffer.
    """
    center_lat = sum(p[0] for p in points) / len(points)
    center_lon = sum(p[1] for p in points) / len(points)
    max_dist   = max(
        _haversine(center_lat, center_lon, p[0], p[1])
        for p in points
    )
    radius = max_dist + buffer
    return center_lat, center_lon, radius

# ── CS Hardware Lab
_HW_POINTS = [
    (7.604364, 5.306518),  # Front
    (7.604451, 5.306510),  # Middle
    (7.604448, 5.306505),  # Back
]
_HW_LAT, _HW_LON, _HW_RAD = _compute_center_radius(
    _HW_POINTS, buffer=5)
_HW_RAD = max(_HW_RAD, 25)

# ── CS Software Lab
_SW_POINTS = [
    (7.604602, 5.306361),  # Front
    (7.604538, 5.306381),  # Middle
    (7.604506, 5.306402),  # Back
]
_SW_LAT, _SW_LON, _SW_RAD = _compute_center_radius(
    _SW_POINTS, buffer=5)
_SW_RAD = max(_SW_RAD, 25)

# ── Print computed values for verification
print(f"  📍 CS Hardware Lab:")
print(f"     Center : ({_HW_LAT:.6f}, "
      f"{_HW_LON:.6f})")
print(f"     Radius : {_HW_RAD:.2f}m")

print(f"  📍 CS Software Lab:")
print(f"     Center : ({_SW_LAT:.6f}, "
      f"{_SW_LON:.6f})")
print(f"     Radius : {_SW_RAD:.2f}m")

# ── Venue registry
VENUES = {
    "CS Hardware Lab": {
        "name"      : "CS Hardware Lab",
        "center_lat": _HW_LAT,
        "center_lon": _HW_LON,
        "radius_m"  : _HW_RAD,
        "points"    : _HW_POINTS
    },
    "CS Software Lab": {
        "name"      : "CS Software Lab",
        "center_lat": _SW_LAT,
        "center_lon": _SW_LON,
        "radius_m"  : _SW_RAD,
        "points"    : _SW_POINTS
    }
}

# ─────────────────────────────────────────────
# VERIFICATION FUNCTIONS
# ─────────────────────────────────────────────

def get_venue_info(venue_name):
    """Get venue config by name."""
    return VENUES.get(venue_name)

def verify_location(lat, lon, venue_name=None,
                    accuracy_m=None):
    """
    Verify student GPS location against venue.

    If venue_name provided: checks against that
    specific venue (course-aware verification).

    If not provided: checks all venues and returns
    closest match.

    accuracy_m: GPS accuracy in meters from device.
    If > 20m, GPS is considered unreliable.

    Returns result dict with verdict + details.
    """
    # GPS accuracy check
    gps_reliable = True
    if accuracy_m is not None and accuracy_m > 20:
        gps_reliable = False

    if venue_name and venue_name in VENUES:
        venues_to_check = [VENUES[venue_name]]
    else:
        venues_to_check = list(VENUES.values())

    best     = None
    best_dist = float('inf')

    for venue in venues_to_check:
        dist = _haversine(
            lat, lon,
            venue["center_lat"],
            venue["center_lon"]
        )
        if dist < best_dist:
            best_dist = dist
            best      = venue

    if not best:
        return {
            "allowed"     : False,
            "verdict"     : "BLOCKED",
            "method"      : "GPS",
            "gps_reliable": gps_reliable,
            "distance_m"  : None,
            "radius_m"    : None,
            "venue_name"  : venue_name,
            "message"     : "Venue not found",
            "timestamp"   : datetime.now().isoformat()
        }

    within_radius = best_dist <= best["radius_m"]
    allowed       = within_radius and gps_reliable

    if not gps_reliable:
        verdict = "GPS_UNRELIABLE"
        message = (
            f"GPS accuracy too low ({accuracy_m:.0f}m). "
            f"Please use QR code verification.")
    elif within_radius:
        verdict = "ALLOWED"
        message = (
            f"✅ Location verified. You are "
            f"{best_dist:.1f}m from "
            f"{best['name']}.")
    else:
        verdict = "BLOCKED"
        message = (
            f"❌ You are {best_dist:.1f}m away "
            f"from {best['name']} "
            f"(allowed: {best['radius_m']:.0f}m). "
            f"Please be physically present.")

    return {
        "allowed"     : allowed,
        "verdict"     : verdict,
        "method"      : "GPS",
        "gps_reliable": gps_reliable,
        "distance_m"  : best_dist,
        "radius_m"    : best["radius_m"],
        "venue_name"  : best["name"],
        "message"     : message,
        "timestamp"   : datetime.now().isoformat()
    }
