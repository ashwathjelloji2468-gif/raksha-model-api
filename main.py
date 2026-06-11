from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np
import math

app = FastAPI(title="RakshaAI Model API", description="FastAPI Server for XGBoost and StandardScaler models")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-phi-seven-18.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_raksha_tuned.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler_raksha.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ── Blood compatibility tables ────────────────────────────────
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

ANTIGEN_WEIGHT = {
    "Kell":   0.20,
    "Duffy":  0.15,
    "Kidd":   0.15,
    "MNS":    0.10,
    "Lewis":  0.08,
}

class DonorData(BaseModel):
    cycle_of_donations: int
    donations_till_date: int
    total_calls: int
    frequency_in_days: int
    registration_date: str
    last_donation_date: str
    last_contacted_date: str

class PatientQuery(BaseModel):
    blood_group: str
    rh_factor: str
    antibody_history: List[str] = []
    city: str
    latitude: float
    longitude: float
    transfusion_count: int
    last_transfusion_date: str
    diagnosis: str

class DonorRecord(BaseModel):
    donor_id: str
    blood_group: str
    city: str
    latitude: float
    longitude: float
    cycle_of_donations: float
    donations_till_date: float
    total_calls: float
    frequency_in_days: float
    registration_date: str
    last_donation_date: str
    last_contacted_date: str
    has_conditions: bool = False
    feedback_score: Optional[float] = 5.0
    extended_antigens: Optional[List[str]] = []

class MatchRequest(BaseModel):
    patient: PatientQuery
    donors: List[DonorRecord]

class SinglePredictRequest(BaseModel):
    data: Dict[str, Any]

class BatchPredictRequest(BaseModel):
    inputs: List[Dict[str, Any]]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 1)

def activity_score(donor: DonorRecord) -> float:
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
        dt = pd.to_datetime(val, errors="coerce")
        row[col + "_days"] = (ref - dt).days if pd.notna(dt) else 999

    df = pd.DataFrame([row])
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    prob = float(model.predict_proba(scaler.transform(df))[0][1])
    return round(prob * 100, 2)

def antigen_score(patient_antibodies: List[str],
                  donor_antigens:    List[str]) -> float:
    if not patient_antibodies:
        return 100.0
    risk = 0.0
    for ab in patient_antibodies:
        if ab in donor_antigens:
            risk += ANTIGEN_WEIGHT.get(ab, 0.05)
    return round(max(0, (1 - risk) * 100), 1)

def recency_score(last_donation_date: str) -> float:
    try:
        days = (pd.Timestamp.today() -
                pd.to_datetime(last_donation_date)).days
    except Exception:
        days = 999
    return round(max(0, min(100, (1 - days / 365) * 100)), 1)

def distance_score(dist_km: float) -> float:
    if dist_km <= 5:   return 100.0
    if dist_km <= 10:  return 85.0
    if dist_km <= 20:  return 65.0
    if dist_km <= 50:  return 40.0
    return max(0, round(100 - dist_km * 0.8, 1))

def feedback_score(score: Optional[float]) -> float:
    if score is None: return 70.0
    return round((score / 5.0) * 100, 1)

@app.get("/")
async def root():
    return {"status": "RakshaAI Model API is running"}

@app.post("/predict")
def predict(donor: DonorData):
    donor_dict = donor.dict()
    df_input = pd.DataFrame([donor_dict])

    # Convert dates to days
    reference_date = pd.Timestamp.today()
    date_cols = ['registration_date', 'last_donation_date', 'last_contacted_date']
    for col in date_cols:
        df_input[col] = pd.to_datetime(df_input[col], errors='coerce')
        df_input[col + '_days'] = (reference_date - df_input[col]).dt.days
        df_input.drop(columns=[col], inplace=True)

    trained_columns = scaler.feature_names_in_
    df_input = df_input.reindex(columns=trained_columns, fill_value=0)
    scaled_input = scaler.transform(df_input)

    label_encoded = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1])
    label = 'Active' if label_encoded == 1 else 'Inactive'
    confidence = round(probability * 100, 2)

    # ── Real factor scores derived from actual input values ──
    days_since_last_donation = float(df_input.get(
        'last_donation_date_days', pd.Series([999])
    ).iloc[0])
    days_since_contacted = float(df_input.get(
        'last_contacted_date_days', pd.Series([999])
    ).iloc[0])
    days_registered = float(df_input.get(
        'registration_date_days', pd.Series([0])
    ).iloc[0])

    # Recency score: donated within 90 days = 100%, 180 days = 50%, 365+ = 0%
    recency_score = round(max(0, min(100, (1 - days_since_last_donation / 365) * 100)), 1)

    # Engagement score: contacted recently = higher score
    engagement_score = round(max(0, min(100, (1 - days_since_contacted / 180) * 100)), 1)

    # Loyalty score: how long registered, capped at 3 years
    loyalty_score = round(min(100, (days_registered / 1095) * 100), 1)

    # Donation frequency score based on cycle count
    cycle_score = round(min(100, (donor.cycle_of_donations / 10) * 100), 1)

    return {
        "success": True,
        "prediction": label,
        "confidence": confidence,
        "match_probability": probability,
        "evaluated_at": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S"),
        "factors": {
            "recency": {
                "label": "Last donation",
                "score": recency_score,
                "detail": f"{int(days_since_last_donation)} days ago"
            },
            "engagement": {
                "label": "Last contacted",
                "score": engagement_score,
                "detail": f"{int(days_since_contacted)} days ago"
            },
            "loyalty": {
                "label": "Time registered",
                "score": loyalty_score,
                "detail": f"{int(days_registered // 30)} months"
            },
            "cycle": {
                "label": "Donation cycles",
                "score": cycle_score,
                "detail": f"{donor.cycle_of_donations} cycles completed"
            }
        }
    }

@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    try:
        probabilities = []
        predictions = []
        
        for item in req.inputs:
            # Reindex to match trained features (similar logic to predict_single)
            df = pd.DataFrame([item])
            
            # Simple preprocess for batch
            reference_date = pd.Timestamp.today()
            date_cols = ['registration_date', 'last_donation_date', 'last_contacted_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col + '_days'] = (reference_date - df[col]).dt.days
                    df.drop(columns=[col], inplace=True)
            
            trained_columns = scaler.feature_names_in_
            df = df.reindex(columns=trained_columns, fill_value=0)
            scaled_features = scaler.transform(df)
            
            prob = model.predict_proba(scaled_features)[0, 1]
            pred = int(model.predict(scaled_features)[0])
            probabilities.append(float(prob))
            predictions.append(pred)
            
        return {
            "success": True,
            "probabilities": probabilities,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/match")
def match_donors(req: MatchRequest):
    patient   = req.patient
    results   = []

    for donor in req.donors:

        # ── Hard filter 1: ABO compatibility ──
        compatible_recipients = ABO_COMPAT.get(donor.blood_group, [])
        if patient.blood_group not in compatible_recipients:
            continue

        # ── Hard filter 2: Medical conditions ──
        if donor.has_conditions:
            continue

        # ── Compute distance ──
        dist_km = haversine_km(
            patient.latitude, patient.longitude,
            donor.latitude,   donor.longitude
        )

        # ── Score each dimension ──
        s_activity  = activity_score(donor)
        s_antigen   = antigen_score(patient.antibody_history,
                                    donor.extended_antigens or [])
        s_recency   = recency_score(donor.last_donation_date)
        s_distance  = distance_score(dist_km)
        s_feedback  = feedback_score(donor.feedback_score)

        # ── Weighted composite score ──
        composite = round(
            s_activity  * 0.30 +
            s_antigen   * 0.25 +
            s_recency   * 0.20 +
            s_distance  * 0.15 +
            s_feedback  * 0.10,
        2)

        results.append({
            "donor_id":        donor.donor_id,
            "blood_group":     donor.blood_group,
            "city":            donor.city,
            "distance_km":     dist_km,
            "composite_score": composite,
            "breakdown": {
                "activity_score":  {
                    "score": s_activity,
                    "weight": "30%",
                    "detail": "XGBoost model — likelihood of active donation"
                },
                "antigen_score":   {
                    "score": s_antigen,
                    "weight": "25%",
                    "detail": "Extended antigen compatibility vs patient antibodies"
                },
                "recency_score":   {
                    "score": s_recency,
                    "weight": "20%",
                    "detail": f"Last donated {donor.last_donation_date}"
                },
                "distance_score":  {
                    "score": s_distance,
                    "weight": "15%",
                    "detail": f"{dist_km} km from patient"
                },
                "feedback_score":  {
                    "score": s_feedback,
                    "weight": "10%",
                    "detail": f"Platform rating: {donor.feedback_score}/5"
                },
            },
            "alloimmunization_risk": "Low" if s_antigen > 80 else
                                     "Moderate" if s_antigen > 50 else "High",
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    return {
        "success":       True,
        "patient_query": patient.blood_group,
        "total_matched": len(results),
        "top_matches":   results[:5],
        "evaluated_at":  pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
