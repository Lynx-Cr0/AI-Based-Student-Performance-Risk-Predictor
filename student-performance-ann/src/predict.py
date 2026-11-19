"""Shared inference function for the Streamlit app and FastAPI."""
import os
os.environ.setdefault("KERAS_BACKEND","tensorflow")
from pathlib import Path
import joblib, numpy as np, keras

ROOT=Path(__file__).resolve().parents[1]
ART=joblib.load(ROOT/"models/preprocessing.joblib")
MODEL=keras.models.load_model(ROOT/"models/student_performance.keras")
FEATURES=ART["feature_columns"]

def predict_student(values):
    missing=[f for f in FEATURES if f not in values]
    if missing: raise ValueError("Missing features: "+", ".join(missing))
    row=np.array([[float(values[f]) for f in FEATURES]],dtype=float)
    row=ART["scaler"].transform(ART["imputer"].transform(row))
    probs=MODEL.predict(row,verbose=0)[0]
    idx=int(np.argmax(probs))
    labels=ART["label_encoder"].classes_
    return {"predicted_performance":str(labels[idx]),
            "confidence":float(probs[idx]),
            "probabilities":{str(labels[i]):float(probs[i]) for i in range(len(labels))}}
