import math
from datetime import datetime, timedelta, timezone
from typing import Optional


IST = timezone(timedelta(hours=5, minutes=30))


SERVICE_CENTERS = [
    {
        "name": "AutoFix Delhi",
        "lat": 28.620000,
        "lon": 77.210000,
        "slots_available": 4,
        "eta_minutes": 18,
        "address": "Connaught Place Service Hub, New Delhi",
        "phone": "+91-11-4100-1001",
    },
    {
        "name": "CarCare Noida",
        "lat": 28.570000,
        "lon": 77.320000,
        "slots_available": 6,
        "eta_minutes": 28,
        "address": "Sector 62 Mobility Park, Noida",
        "phone": "+91-120-440-2202",
    },
    {
        "name": "MechPro Gurgaon",
        "lat": 28.460000,
        "lon": 77.030000,
        "slots_available": 3,
        "eta_minutes": 34,
        "address": "Cyber City Auto Lane, Gurugram",
        "phone": "+91-124-430-3303",
    },
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def find_nearest_service(lat: float, lon: float) -> dict:
    nearest = None
    best_distance = float("inf")

    for center in SERVICE_CENTERS:
        dist = haversine_km(lat, lon, center["lat"], center["lon"])
        if dist < best_distance:
            best_distance = dist
            nearest = {
                **center,
                "distance_km": round(dist, 2),
            }

    return nearest if nearest is not None else {}


def _normalize_requested_slot(
    requested_date: Optional[str],
    requested_time: Optional[str],
) -> Optional[datetime]:
    if not requested_date or not requested_time:
        return None

    candidates = [
        f"{requested_date} {requested_time}",
        f"{requested_date} {requested_time} IST",
    ]
    formats = [
        "%d %b %Y %I:%M %p",
        "%d %b %Y %I:%M %p IST",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M %p IST",
    ]

    for candidate in candidates:
        for fmt in formats:
            try:
                parsed = datetime.strptime(candidate.strip(), fmt)
                return parsed.replace(tzinfo=IST)
            except ValueError:
                continue
    return None


def build_service_schedule(
    center: dict,
    urgency: str,
    requested_date: Optional[str] = None,
    requested_time: Optional[str] = None,
) -> dict:
    now = datetime.now(IST)
    offset_minutes = 45 if urgency == "HIGH" else 120
    scheduled_at = now + timedelta(minutes=center.get("eta_minutes", 20) + offset_minutes)

    requested_slot = _normalize_requested_slot(requested_date, requested_time)
    if requested_slot is not None:
        min_allowed = now + timedelta(minutes=max(30, center.get("eta_minutes", 20)))
        scheduled_at = max(requested_slot, min_allowed)

    return {
        "scheduled_at": scheduled_at.isoformat(),
        "scheduled_date": scheduled_at.strftime("%d %b %Y"),
        "scheduled_time": scheduled_at.strftime("%I:%M %p IST"),
    }
