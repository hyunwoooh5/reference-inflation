from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
import pickle
from predict import Paper, predict_single
from util import *


class PredictResponse(BaseModel):
    number_of_references: float


app = FastAPI(title="reference-inflation")


try:
    with open("bin/model.bin", "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    with open("model.bin", "rb") as f:
        model = pickle.load(f)


@app.post("/predict")
def predict(paper: Paper) -> PredictResponse:
    prob = predict_single(model, paper)

    return PredictResponse(
        number_of_references=prob
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
