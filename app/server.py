# app/server.py
import os
from typing import List, Literal, Optional

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "iris-classifier")
DEFAULT_VERSION     = os.getenv("MODEL_VERSION", "1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# cache of loaded versions -> model objects
_model_cache: dict[str, object] = {}
_current_version: str = DEFAULT_VERSION

IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


def _model_uri_for(version: str) -> str:
    return f"models:/{MODEL_NAME}/{version}"


def _load_model(version: str):
    """Load a model version, caching it for reuse."""
    if version in _model_cache:
        return _model_cache[version]
    try:
        model = mlflow.pyfunc.load_model(_model_uri_for(version))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Unable to load model {MODEL_NAME} v{version}: {e}")
    _model_cache[version] = model
    return model


# Preload the default version at startup
_default_model = _load_model(DEFAULT_VERSION)

# ----- Pydantic schemas -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., description="Sepal length (cm)", examples=[5.1])
    sepal_width:  float = Field(..., description="Sepal width (cm)", examples=[3.5])
    petal_length: float = Field(..., description="Petal length (cm)", examples=[1.4])
    petal_width:  float = Field(..., description="Petal width (cm)", examples=[0.2])


class PredictRequest(BaseModel):
    samples: List[IrisSample] = Field(
        ...,
        description="List of Iris samples with sepal/petal measurements (cm)."
    )


class PredictResponse(BaseModel):
    class_id: List[int]
    class_label: List[str]


class VersionResponse(BaseModel):
    model_name: str
    version: str


class UpdateVersionRequest(BaseModel):
    version: str = Field(..., description="Target registered model version (as a string, e.g., '2').")


app = FastAPI(
    title="Iris Classifier API",
    description="Serve a registered MLflow model to predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0"
)


@app.get("/", tags=["system"], summary="Health check")
async def health():
    return {"status": "ok", "model_name": MODEL_NAME, "version": _current_version}


@app.get("/version", response_model=VersionResponse, tags=["model"], summary="Endpoint to view the current served version")
async def get_version():
    return VersionResponse(model_name=MODEL_NAME, version=_current_version)


@app.put("/version", response_model=VersionResponse, tags=["model"], summary="Endpoint to allow us to select a version to serve")
async def select_version(req: UpdateVersionRequest):
    global _current_version
    # attempt to load first to validate availability
    _load_model(req.version)
    _current_version = req.version
    return VersionResponse(model_name=MODEL_NAME, version=_current_version)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
async def predict(req: PredictRequest) -> PredictResponse:
    # Predict using the correct served version
    model = _load_model(_current_version)

    # Convert incoming structured samples to a numeric array of shape (n, 4)
    X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width] for s in req.samples], dtype=float)
    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Normalize output to 1D int list
    preds = np.asarray(preds).reshape(-1).astype(int).tolist()
    labels = [IRIS_LABELS.get(cid, str(cid)) for cid in preds]
    return PredictResponse(class_id=preds, class_label=labels)
