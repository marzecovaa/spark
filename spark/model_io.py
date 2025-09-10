import os
import time
import joblib
import glob
from tensorflow import keras

from spark.params import LOCAL_REGISTRY_PATH, MODEL_TYPE

def save_model(model) -> str:
    """Save model locally with timestamp and format based on MODEL_TYPE."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ext = ".joblib" if MODEL_TYPE == "joblib" else ".h5"
    filename = f"model_{timestamp}{ext}"
    model_path = os.path.join(LOCAL_REGISTRY_PATH, filename)

    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    if MODEL_TYPE == "joblib":
        joblib.dump(model, model_path)
    else:
        model.save(model_path)

    print(f"Saved locally: {model_path}")
    return model_path


def save_model_combined(model) -> str:
    """Save model locally with timestamp and format based on MODEL_TYPE."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ext = ".joblib" if MODEL_TYPE == "joblib" else ".h5"
    filename = f"model_combined_{timestamp}{ext}"
    model_path = os.path.join(LOCAL_REGISTRY_PATH, filename)

    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    if MODEL_TYPE == "joblib":
        joblib.dump(model, model_path)
    else:
        model.save(model_path)

    print(f"Saved locally: {model_path}")
    return model_path


def load_model(path: str = None):
    """Load model from given path or fallback to latest in registry."""
    if path:
        print(f"Loading model from: {path}")
        return joblib.load(path) if path.endswith(".joblib") else keras.models.load_model(path)

    paths = glob.glob(os.path.join(LOCAL_REGISTRY_PATH, "model_*.joblib"))
    if not paths:
        print("No models found in local registry.")
        return None

    latest_path = max(paths, key=os.path.getctime)
    print(f"Loading model from: {latest_path}")
    return joblib.load(latest_path) if latest_path.endswith(".joblib") else keras.models.load_model(latest_path)


def save_transformer(transformer) -> str:
    """Save fitted transformer separately for inference use."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"transformer_{timestamp}.joblib"
    path = os.path.join(LOCAL_REGISTRY_PATH, filename)

    os.makedirs(LOCAL_REGISTRY_PATH, exist_ok=True)
    joblib.dump(transformer, path)

    print(f"✅ Transformer saved at: {path}")
    return path


def load_latest_transformer():
    """ loads latest transformer"""
    paths = glob.glob(os.path.join(LOCAL_REGISTRY_PATH, "transformer_*.joblib"))
    if not paths:
        print("No transformer found.")
        return None
    latest_path = max(paths, key=os.path.getctime)
    return joblib.load(latest_path)
