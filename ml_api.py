from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("surveillance_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class InputData(BaseModel):
    encrypted_app_usage_hours :  float 
    social_media_threat_score :  float 
    camera_activation_events: int    
    vpn_usage: int
    microphone_activation_events: int
    call_count_with_suspects: int   
    account_login_failures: int   
    suspicious_contacts_count: int   
    device_changes_last_30_days:  int
    night_movements_hours: float  

# Home route
@app.get("/")
def home():
    return {"message": "Surveillance Model API is running"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([
        data.encrypted_app_usage_hours, 
        data.social_media_threat_score, 
        data.camera_activation_events,    
        data.vpn_usage,
        data.microphone_activation_events,
        data.call_count_with_suspects,   
        data.account_login_failures,   
        data.suspicious_contacts_count,   
        data.device_changes_last_30_days,
        data.night_movements_hours
        ]).reshape(1, -1)
    
    prediction = model.predict(input_array)

    return {"prediction": prediction.tolist()}

    

