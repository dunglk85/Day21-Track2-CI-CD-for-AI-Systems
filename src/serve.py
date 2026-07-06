from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os
import uvicorn

app = FastAPI()

# Doc cau hinh tu bien moi truong (fail fast neu thieu)
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

# Dam bao thu muc ton tai
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_model():
    """Tai file model.pkl tu GCS ve may khi server khoi dong."""
    try:
        print(f"Dang tai mo hinh tu bucket: {GCS_BUCKET}...")
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Tai mo hinh thanh cong ve: {MODEL_PATH}")
    except Exception as e:
        print(f"Loi khi tai mo hinh: {e}")
        if not os.path.exists(MODEL_PATH):
             print("Canh bao: Khong tim thay mo hinh tai local.")

# Goi ham tai mo hinh khi khoi dong
download_model()

# Load mo hinh vao bo nho
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Canh bao: Server khoi chay ma khong co mo hinh.")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.
    Nhan 12 dac trung va tra ve nhan: thap, trung_binh, cao.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    # Du doan
    prediction = int(model.predict([req.features])[0])
    
    # Anh xa nhan theo project-context.md
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = labels.get(prediction, "unknown")

    return {
        "prediction": prediction,
        "label": label
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
