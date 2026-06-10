from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import joblib
import pickle
import pandas as pd
import numpy as np

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

# Global variables for models
model = None
scaler = None

# Feature names expected by the scaler (62 columns)
FEATURE_COLUMNS = [
    'role_status', 'bridge_status', 'latitude', 'longitude', 'quantity_required', 
    'donations_till_date', 'cycle_of_donations', 'total_calls', 'frequency_in_days', 
    'status_of_bridge', 'registration_date_days', 'last_contacted_date_days', 
    'last_donation_date_days', 'last_transfusion_date_days', 'expected_next_transfusion_date_days', 
    'next_eligible_date_days', 'last_bridge_donation_date_days', 
    'role_Bridge Donor', 'role_Emergency Donor', 'role_Guest', 'role_Patient', 'role_Volunteer', 
    'blood_group_A Negative', 'blood_group_A Positive', 'blood_group_A1 Positive', 
    'blood_group_A1B Positive', 'blood_group_A2 Negative', 'blood_group_A2 Positive', 
    'blood_group_A2B Negative', 'blood_group_A2B Positive', 'blood_group_AB Negative', 
    'blood_group_AB Positive', 'blood_group_B Negative', 'blood_group_B Positive', 
    'blood_group_Bombay Blood Group', 'blood_group_Do not Know', 'blood_group_O Negative', 
    'blood_group_O Positive', 'blood_group_Unknown', 
    'gender_Female', 'gender_Male', 'gender_Non-binary', 'gender_Other', 
    'gender_Prefer not to say', 'gender_Unknown', 
    'donor_type_One-Time Donor', 'donor_type_Other', 'donor_type_Regular Donor', 
    'bridge_gender_Female', 'bridge_gender_Male', 'bridge_gender_Unknown', 
    'bridge_blood_group_A Negative', 'bridge_blood_group_A Positive', 
    'bridge_blood_group_AB Negative', 'bridge_blood_group_AB Positive', 
    'bridge_blood_group_B Negative', 'bridge_blood_group_B Positive', 
    'bridge_blood_group_O Negative', 'bridge_blood_group_O Positive', 
    'bridge_blood_group_Unknown', 'eligibility_status_eligible', 
    'eligibility_status_not eligible'
]

# Categorical mapping fields for automatic one-hot encoding helper
CATEGORICAL_FIELDS = {
    'role': ['Bridge Donor', 'Emergency Donor', 'Guest', 'Patient', 'Volunteer'],
    'blood_group': [
        'A Negative', 'A Positive', 'A1 Positive', 'A1B Positive', 'A2 Negative', 
        'A2 Positive', 'A2B Negative', 'A2B Positive', 'AB Negative', 'AB Positive', 
        'B Negative', 'B Positive', 'Bombay Blood Group', 'Do not Know', 'O Negative', 
        'O Positive', 'Unknown'
    ],
    'gender': ['Female', 'Male', 'Non-binary', 'Other', 'Prefer not to say', 'Unknown'],
    'donor_type': ['One-Time Donor', 'Other', 'Regular Donor'],
    'bridge_gender': ['Female', 'Male', 'Unknown'],
    'bridge_blood_group': [
        'A Negative', 'A Positive', 'AB Negative', 'AB Positive', 'B Negative', 
        'B Positive', 'O Negative', 'O Positive', 'Unknown'
    ],
    'eligibility_status': ['eligible', 'not eligible']
}

@app.on_event("startup")
def load_models():
    global model, scaler
    model_path = os.path.join(os.path.dirname(__file__), "model", "xgboost_raksha_tuned.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler_raksha.pkl")
    
    print("Loading models...")
    # Load Scaler
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully via joblib.")
    except Exception as e:
        print(f"joblib load failed for scaler, trying pickle: {e}")
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully via pickle.")
        except Exception as ex:
            print(f"Error loading scaler: {ex}")
            
    # Load XGBoost Model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("XGBoost model loaded successfully.")
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")

@app.get("/")
async def root():
    return {"status": "RakshaAI Model API is running"}

def preprocess_input(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms raw dictionary input containing numerical and categorical fields
    into the exact 62-column format expected by the StandardScaler and XGBoost model.
    """
    # 1. Initialize all feature columns to 0.0 or default values
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    
    # 2. Extract numerical features (default to 0.0 if not provided)
    numerical_cols = [
        'role_status', 'bridge_status', 'latitude', 'longitude', 'quantity_required', 
        'donations_till_date', 'cycle_of_donations', 'total_calls', 'frequency_in_days', 
        'status_of_bridge', 'registration_date_days', 'last_contacted_date_days', 
        'last_donation_date_days', 'last_transfusion_date_days', 'expected_next_transfusion_date_days', 
        'next_eligible_date_days', 'last_bridge_donation_date_days'
    ]
    for col in numerical_cols:
        features[col] = float(raw_data.get(col, 0.0))
        
    # 3. Apply categorical one-hot encoding
    for field, categories in CATEGORICAL_FIELDS.items():
        val = raw_data.get(field)
        if val is not None:
            # Match categorical value to standard names
            val_str = str(val).strip()
            # If the user sends shortcodes like "O+", convert to standard O Positive
            if field in ['blood_group', 'bridge_blood_group']:
                normalization = {
                    "O+": "O Positive", "O-": "O Negative",
                    "A+": "A Positive", "A-": "A Negative",
                    "B+": "B Positive", "B-": "B Negative",
                    "AB+": "AB Positive", "AB-": "AB Negative"
                }
                val_str = normalization.get(val_str, val_str)
                
            # One-hot encoding mapping
            for cat in categories:
                col_name = f"{field}_{cat}"
                if col_name in features:
                    features[col_name] = 1.0 if val_str.lower() == cat.lower() else 0.0
                    
    # 4. Create DataFrame and enforce column order matching FEATURE_COLUMNS
    df = pd.DataFrame([features])
    df = df[FEATURE_COLUMNS]
    return df

class SinglePredictRequest(BaseModel):
    data: Dict[str, Any]

class BatchPredictRequest(BaseModel):
    inputs: List[Dict[str, Any]]

@app.post("/predict")
async def predict_single(req: Dict[str, Any]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models are not loaded or unpickling failed.")
    try:
        # Support both {"data": {...}} wrapper and direct {...} formats
        input_data = req.get("data", req) if isinstance(req, dict) else req
        
        # Preprocess
        df = preprocess_input(input_data)
        
        # Scale
        scaled_features = scaler.transform(df)
        
        # Predict probability of positive match/acceptance class (class 1)
        prob = model.predict_proba(scaled_features)[0, 1]
        prediction = int(model.predict(scaled_features)[0])
        
        # Map class 0/1 to Active/Inactive or similar if needed, or return prediction directly
        # Let's check: in the user's example, they say:
        # Returns { prediction: "Active", confidence: 97.32 }
        # Let's map prediction: 1 -> "Active", 0 -> "Inactive"
        prediction_label = "Active" if prediction == 1 else "Inactive"
        confidence = float(prob) * 100.0 if prediction == 1 else (1.0 - float(prob)) * 100.0
        
        return {
            "success": True,
            "prediction": prediction_label,
            "confidence": round(confidence, 2),
            "match_probability": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models are not loaded or unpickling failed.")
    try:
        probabilities = []
        predictions = []
        
        for item in req.inputs:
            df = preprocess_input(item)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
