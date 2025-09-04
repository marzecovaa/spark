from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from spark.model_io import load_model, load_latest_transformer

app = FastAPI()

# CORS setup for dev/demo use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
@app.on_event("startup")
def load_model_on_startup():
    app.state.model = load_model()
    app.state.transformer = load_latest_transformer()


# Input schema for smartwatch-based Parkinson’s detection
class ParkinsonsInput(BaseModel):
    age: int
    age_at_diagnosis: int
    height: float
    weight: float
    bmi: float
    handedness: str
    appearance_in_kinship: str
    subject_id: int
    gender: str

    # Questionnaire responses: fields "1" to "30"
    q1: bool = Field(..., alias="01")
    q2: bool = Field(..., alias="02")
    q3: bool = Field(..., alias="03")
    q4: bool = Field(..., alias="04")
    q5: bool = Field(..., alias="05")
    q6: bool = Field(..., alias="06")
    q7: bool = Field(..., alias="07")
    q8: bool = Field(..., alias="08")
    q9: bool = Field(..., alias="09")
    q10: bool = Field(..., alias="10")
    q11: bool = Field(..., alias="11")
    q12: bool = Field(..., alias="12")
    q13: bool = Field(..., alias="13")
    q14: bool = Field(..., alias="14")
    q15: bool = Field(..., alias="15")
    q16: bool = Field(..., alias="16")
    q17: bool = Field(..., alias="17")
    q18: bool = Field(..., alias="18")
    q19: bool = Field(..., alias="19")
    q20: bool = Field(..., alias="20")
    q21: bool = Field(..., alias="21")
    q22: bool = Field(..., alias="22")
    q23: bool = Field(..., alias="23")
    q24: bool = Field(..., alias="24")
    q25: bool = Field(..., alias="25")
    q26: bool = Field(..., alias="26")
    q27: bool = Field(..., alias="27")
    q28: bool = Field(..., alias="28")
    q29: bool = Field(..., alias="29")
    q30: bool = Field(..., alias="30")

    class Config:
        allow_population_by_field_name = True

@app.post("/predict")
def predict(data: ParkinsonsInput):
    model = app.state.model
    if model is None:
        return {"error": "Model not loaded"}

    # Convert input to DataFrame
    input_dict = data.model_dump(by_alias=True)
    X_raw = pd.DataFrame([input_dict])

    try:
        X_processed = app.state.transformer.transform(X_raw)
        y_pred = model.predict(X_processed)

        result = {"prediction": float(y_pred[0])}

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_processed)[0]
            for i, prob in enumerate(y_proba):
                result[f"prob_class_{i}"] = float(prob)

        return result

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/")
def root():
    return {"greeting": "Hello! Welcome and wish you good health ❤️"}
