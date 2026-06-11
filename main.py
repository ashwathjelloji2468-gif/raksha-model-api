from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import math, joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model  = joblib.load('model/xgboost_raksha_tuned.pkl')
scaler = joblib.load('model/scaler_raksha.pkl')

# ── City coordinates (India) ──────────────────────────────────
CITY_COORDS = {
    "hyderabad":     (17.3850, 78.4867),
    "secunderabad":  (17.4399, 78.4983),
    "warangal":      (17.9784, 79.5941),
    "mumbai":        (19.0760, 72.8777),
    "pune":          (18.5204, 73.8567),
    "bangalore":     (12.9716, 77.5946),
    "bengaluru":     (12.9716, 77.5946),
    "chennai":       (13.0827, 80.2707),
    "delhi":         (28.7041, 77.1025),
    "kolkata":       (22.5726, 88.3639),
    "ahmedabad":     (23.0225, 72.5714),
    "jaipur":        (26.9124, 75.7873),
    "lucknow":       (26.8467, 80.9462),
    "nagpur":        (21.1458, 79.0882),
    "visakhapatnam": (17.6868, 83.2185),
    "vijayawada":    (16.5062, 80.6480),
    "karimnagar":    (18.4386, 79.1288),
    "nizamabad":     (18.6725, 78.0941),
}

def get_coords(city: str, lat: float, lon: float):
    if lat and lon and lat != 0 and lon != 0:
        return lat, lon
    key = city.lower().strip() if city else ""
    return CITY_COORDS.get(key, (17.3850, 78.4867))

# ── Blood compatibility ───────────────────────────────────────
ABO_COMPAT = {
    "O-":  ["O-","O+","A-","A+","B-","B+","AB-","AB+"],
    "O+":  ["O+","A+","B+","AB+"],
    "A-":  ["A-","A+","AB-","AB+"],
    "A+":  ["A+","AB+"],
    "B-":  ["B-","B+","AB-","AB+"],
    "B+":  ["B+","AB+"],
    "AB-": ["AB-","AB+"],
    "AB+": ["AB+"],
}

ANTIGEN_RISK_WEIGHT = {
    "Kell":   0.30,
    "Anti-D": 0.25,
    "Anti-E": 0.20,
    "Anti-C": 0.20,
    "Duffy":  0.15,
    "Kidd":   0.15,
    "MNS":    0.10,
    "Lewis":  0.08,
}

class PatientQuery(BaseModel):
    blood_group:           str
    rh_factor:             str = ""
    antibody_history:      List[str] = []
    city:                  str = ""
    latitude:              float = 0
    longitude:             float = 0
    transfusion_count:     int = 0
    last_transfusion_date: str = ""
    diagnosis:             str = "Thalassemia"

class DonorRecord(BaseModel):
    donor_id:            str
    donor_name:          str = ""
    blood_group:         str
    city:                str = ""
    latitude:            float = 0
    longitude:           float = 0
    cycle_of_donations:  float = 0
    donations_till_date: float = 0
    total_calls:         float = 0
    frequency_in_days:   float = 90
    registration_date:   str = "2023-01-01"
    last_donation_date:  str = "2024-01-01"
    last_contacted_date: str = "2024-01-01"
    has_conditions:      bool = False
    feedback_score:      Optional[float] = 5.0
    extended_antigens:   Optional[List[str]] = []

class DonorOnly(BaseModel):
    donor_id:            str
    blood_group:         str
    city:                str = ""
    latitude:            float = 0
    longitude:           float = 0
    cycle_of_donations:  float = 0
    donations_till_date: float = 0
    total_calls:         float = 0
    frequency_in_days:   float = 90
    registration_date:   str = "2023-01-01"
    last_donation_date:  str = "2024-01-01"
    last_contacted_date: str = "2024-01-01"
    has_conditions:      bool = False
    feedback_score:      Optional[float] = 5.0

class MatchRequest(BaseModel):
    patient: PatientQuery
    donors:  List[DonorRecord]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 1)

def compute_activity_score(donor: DonorRecord) -> tuple:
    ref = pd.Timestamp.today()
    row = {
        "cycle_of_donations":  donor.cycle_of_donations,
        "donations_till_date": donor.donations_till_date,
        "total_calls":         donor.total_calls,
        "frequency_in_days":   donor.frequency_in_days,
    }
    for col, val in [
        ("registration_date",   donor.registration_date),
        ("last_donation_date",  donor.last_donation_date),
        ("last_contacted_date", donor.last_contacted_date),
    ]:
        try:
            dt = pd.to_datetime(val, errors="coerce")
            row[col + "_days"] = int((ref - dt).days) if pd.notna(dt) else 999
        except:
            row[col + "_days"] = 999
    df = pd.DataFrame([row])
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    prob = float(model.predict_proba(scaler.transform(df))[0][1])
    label = "Active" if prob >= 0.5 else "Inactive"
    return round(prob * 100, 1), label

def compute_antigen_score(patient_abs: List[str],
                           donor_antigens: List[str]) -> tuple:
    if not patient_abs:
        return 100.0, "None detected", "Low"
    risk_total = 0.0
    conflicts  = []
    for ab in patient_abs:
        if ab in (donor_antigens or []):
            w = ANTIGEN_RISK_WEIGHT.get(ab, 0.05)
            risk_total += w
            conflicts.append(ab)
    score = round(max(0, (1 - min(risk_total, 1)) * 100), 1)
    if score >= 85:
        risk_level = "Low"
    elif score >= 60:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    detail = ("No antibody conflicts" if not conflicts
              else f"Conflicts: {', '.join(conflicts)}")
    return score, detail, risk_level

def compute_recency_score(last_donation_date: str) -> tuple:
    try:
        days = (pd.Timestamp.today() -
                pd.to_datetime(last_donation_date)).days
    except:
        days = 999
    score = round(max(0, min(100, (1 - days / 365) * 100)), 1)
    if days < 30:
        detail = f"Donated {days} days ago"
    elif days < 90:
        detail = f"Donated ~{days // 30} months ago"
    else:
        detail = f"Last donation {days} days ago"
    return score, detail

def compute_distance_score(dist_km: float) -> tuple:
    if dist_km <= 5:
        return 100.0, f"{dist_km} km — same area"
    elif dist_km <= 10:
        return 85.0,  f"{dist_km} km — nearby"
    elif dist_km <= 20:
        return 65.0,  f"{dist_km} km — same city"
    elif dist_km <= 50:
        return 40.0,  f"{dist_km} km — short drive"
    else:
        s = round(max(0, 100 - dist_km * 0.8), 1)
        return s, f"{dist_km} km — distant"

def compute_feedback_score(score: Optional[float]) -> tuple:
    if score is None:
        return 70.0, "No ratings yet"
    s = round((score / 5.0) * 100, 1)
    stars = "★" * int(score) + "☆" * (5 - int(score))
    return s, f"{stars} ({score}/5 platform rating)"

@app.get("/")
def root():
    return {"status": "RakshaAI Model API is running"}

@app.post("/predict")
def predict(donor: DonorOnly):
    row = {
        "cycle_of_donations":  donor.cycle_of_donations,
        "donations_till_date": donor.donations_till_date,
        "total_calls":         donor.total_calls,
        "frequency_in_days":   donor.frequency_in_days,
    }
    ref = pd.Timestamp.today()
    for col, val in [
        ("registration_date",   donor.registration_date),
        ("last_donation_date",  donor.last_donation_date),
        ("last_contacted_date", donor.last_contacted_date),
    ]:
        try:
            dt  = pd.to_datetime(val, errors="coerce")
            row[col + "_days"] = int((ref - dt).days) if pd.notna(dt) else 999
        except:
            row[col + "_days"] = 999

    df   = pd.DataFrame([row])
    df   = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    prob = float(model.predict_proba(scaler.transform(df))[0][1])
    conf = round(prob * 100, 2)
    label = "Active" if prob >= 0.5 else "Inactive"

    days_since_donation  = row.get("last_donation_date_days",  999)
    days_since_contact   = row.get("last_contacted_date_days", 999)
    days_registered      = row.get("registration_date_days",   0)

    return {
        "success":          True,
        "prediction":       label,
        "confidence":       conf,
        "match_probability": prob,
        "evaluated_at":     ref.strftime("%Y-%m-%d %H:%M:%S"),
        "factors": {
            "recency": {
                "label":  "Last donation",
                "score":  round(max(0, min(100, (1-days_since_donation/365)*100)), 1),
                "detail": f"{days_since_donation} days ago"
            },
            "engagement": {
                "label":  "Last contacted",
                "score":  round(max(0, min(100, (1-days_since_contact/180)*100)), 1),
                "detail": f"{days_since_contact} days ago"
            },
            "loyalty": {
                "label":  "Time registered",
                "score":  round(min(100, (days_registered/1095)*100), 1),
                "detail": f"{days_registered // 30} months on platform"
            },
            "cycle": {
                "label":  "Donation cycles",
                "score":  round(min(100, (donor.cycle_of_donations/10)*100), 1),
                "detail": f"{int(donor.cycle_of_donations)} cycles completed"
            },
        }
    }

@app.post("/match")
def match_donors(req: MatchRequest):
    patient  = req.patient
    p_lat, p_lon = get_coords(patient.city,
                               patient.latitude,
                               patient.longitude)
    results  = []
    excluded = {"abo_fail": 0, "conditions": 0, "inactive": 0}

    for donor in req.donors:
        # Hard filter 1: ABO compatibility
        compatible = ABO_COMPAT.get(donor.blood_group, [])
        if patient.blood_group not in compatible:
            excluded["abo_fail"] += 1
            continue

        # Hard filter 2: Medical conditions
        if donor.has_conditions:
            excluded["conditions"] += 1
            continue

        # Real coordinates
        d_lat, d_lon = get_coords(donor.city,
                                   donor.latitude,
                                   donor.longitude)
        dist_km = haversine_km(p_lat, p_lon, d_lat, d_lon)

        # All scores computed from real data
        s_activity, activity_label = compute_activity_score(donor)
        s_antigen,  antigen_detail, allo_risk = compute_antigen_score(
            patient.antibody_history, donor.extended_antigens or [])
        s_recency,  recency_detail  = compute_recency_score(donor.last_donation_date)
        s_distance, distance_detail = compute_distance_score(dist_km)
        s_feedback, feedback_detail = compute_feedback_score(donor.feedback_score)

        # Filter inactive donors
        if s_activity < 50:
            excluded["inactive"] += 1
            continue

        # Weighted composite
        composite = round(
            s_activity  * 0.30 +
            s_antigen   * 0.25 +
            s_recency   * 0.20 +
            s_distance  * 0.15 +
            s_feedback  * 0.10, 1)

        results.append({
            "donor_id":    donor.donor_id,
            "donor_name":  donor.donor_name,
            "blood_group": donor.blood_group,
            "city":        donor.city,
            "distance_km": dist_km,
            "composite_score": composite,
            "activity_label":  activity_label,
            "alloimmunization_risk": allo_risk,
            "breakdown": {
                "XGBoost activity": {
                    "score":  s_activity,
                    "weight": "30%",
                    "detail": f"Model prediction: {activity_label}"
                },
                "Antigen compatibility": {
                    "score":  s_antigen,
                    "weight": "25%",
                    "detail": antigen_detail
                },
                "Donation recency": {
                    "score":  s_recency,
                    "weight": "20%",
                    "detail": recency_detail
                },
                "Proximity": {
                    "score":  s_distance,
                    "weight": "15%",
                    "detail": distance_detail
                },
                "Platform feedback": {
                    "score":  s_feedback,
                    "weight": "10%",
                    "detail": feedback_detail
                },
            }
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    return {
        "success":             True,
        "patient_blood_group": patient.blood_group,
        "patient_city":        patient.city,
        "total_scanned":       len(req.donors),
        "total_matched":       len(results),
        "excluded_abo":        excluded["abo_fail"],
        "excluded_conditions": excluded["conditions"],
        "excluded_inactive":   excluded["inactive"],
        "top_matches":         results,
        "evaluated_at":        pd.Timestamp.today().strftime("%d %b %Y · %I:%M %p IST"),
    }
