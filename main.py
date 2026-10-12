import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np

app = FastAPI(title="Penguins ML App")
templates = Jinja2Templates(directory="templates")

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class LoadedClassificationNet:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.mn = np.array(data.get("mn", data.get("x_min")))
        self.mx = np.array(data.get("mx", data.get("x_max")))
        self.names = data.get("names", data.get("classes"))

        if "hidden" in data and isinstance(data["hidden"], list) and len(data["hidden"]) > 0 and isinstance(data["hidden"][0], dict):
            self.hidden_weights = [np.array(n["weights"]) for n in data["hidden"]]
            self.hidden_biases = [n["bias"] for n in data["hidden"]]
            self.output_weights = [np.array(n["weights"]) for n in data["output"]]
            self.output_biases = [n["bias"] for n in data["output"]]
        else:
            self.hidden_weights = [np.array(w) for w in data["hidden_weights"]]
            self.hidden_biases = data["hidden_bias"]
            self.output_weights = [np.array(w) for w in data["output_weights"]]
            self.output_biases = data["output_bias"]

    def predict(self, raw_features: list[float]) -> str:
        x = (np.array(raw_features) - self.mn) / (self.mx - self.mn)

        hidden_outs = np.array([
            sigmoid(np.dot(x, w) + b)
            for w, b in zip(self.hidden_weights, self.hidden_biases)
        ])

        final_outs = np.array([
            sigmoid(np.dot(hidden_outs, w) + b)
            for w, b in zip(self.output_weights, self.output_biases)
        ])

        idx = int(np.argmax(final_outs))
        return self.names[idx]


class LoadedRegressionNet:
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Поддержка ключей mn_x / mx_x (как в вашем файле) и альтернативных
        mn = data.get("mn_x") or data.get("mn") or data.get("x_min")
        mx = data.get("mx_x") or data.get("mx") or data.get("x_max")
        mn_y = data.get("mn_y") or data.get("y_min")
        mx_y = data.get("mx_y") or data.get("y_max")

        self.mn = np.array(mn, dtype=float)
        self.mx = np.array(mx, dtype=float)
        self.mn_y = float(mn_y)
        self.mx_y = float(mx_y)
        self.unit = data.get("unit", "г")

        # Скрытый слой
        self.hidden_weights = [np.array(n["weights"], dtype=float) for n in data["hidden"]]
        self.hidden_biases = [float(n["bias"]) for n in data["hidden"]]

        # Выходной нейрон
        self.out_w = np.array(data["output"]["weights"], dtype=float).flatten()
        self.out_b = float(data["output"]["bias"])

    def predict(self, raw_features: list[float]) -> tuple[float, str]:
        x = (np.array(raw_features, dtype=float) - self.mn) / (self.mx - self.mn)

        hidden_outs = np.array([
            sigmoid(np.dot(x, w) + b)
            for w, b in zip(self.hidden_weights, self.hidden_biases)
        ])

        norm_output = float(np.dot(hidden_outs, self.out_w) + self.out_b)
        real_value = norm_output * (self.mx_y - self.mn_y) + self.mn_y
        return round(real_value, 1), self.unit


BASE_DIR = Path(__file__).resolve().parent
cls_model = LoadedClassificationNet(str(BASE_DIR / "penguins_classification.json"))
reg_model = LoadedRegressionNet(str(BASE_DIR / "penguins_regression.json"))


@app.get("/", response_class=HTMLResponse)
@app.get("/classification", response_class=HTMLResponse)
async def classification_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={"prediction": None}
    )


@app.post("/classification", response_class=HTMLResponse)
async def classification_post(
    request: Request,
    bill_length: Annotated[float, Form()],
    bill_depth: Annotated[float, Form()],
    flipper_length: Annotated[float, Form()],
    body_mass: Annotated[float, Form()]
):
    inputs = [bill_length, bill_depth, flipper_length, body_mass]
    species_name = cls_model.predict(inputs)

    return templates.TemplateResponse(
        request=request,
        name="classification.html",
        context={
            "prediction": species_name,
            "values": {
                "bill_length": bill_length,
                "bill_depth": bill_depth,
                "flipper_length": flipper_length,
                "body_mass": body_mass
            }
        }
    )


@app.get("/regression", response_class=HTMLResponse)
async def regression_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={"prediction": None}
    )


@app.post("/regression", response_class=HTMLResponse)
async def regression_post(
    request: Request,
    bill_length: Annotated[float, Form()],
    bill_depth: Annotated[float, Form()],
    flipper_length: Annotated[float, Form()]
):
    inputs = [bill_length, bill_depth, flipper_length]
    predicted_val, unit = reg_model.predict(inputs)

    return templates.TemplateResponse(
        request=request,
        name="regression.html",
        context={
            "prediction": predicted_val,
            "unit": unit,
            "values": {
                "bill_length": bill_length,
                "bill_depth": bill_depth,
                "flipper_length": flipper_length
            }
        }
    )