import joblib
import numpy as np
import pandas as pd

import io

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fastapi.responses import RedirectResponce
from fasttapi.staticfiles import StaticFiles

from pydantic import BaseModel

import inference
import cnn_ae 

app = FastAPI()

app.mount("/static", StaticFiles(directory="public"))

filename = './model/model.pkl'
loaded_model = joblib.load(filename)

stscfilename = './model/stsc.pkl'
stsc = joblib.load(stscfilename)

class Model(BaseModel):
    X: list[str]

@app.get("/")
def read_root():
    return RedirectResponce('./static/index.html')

@app.post("/predict")
def predict_model(model: Model, ucl: float):
    string_data = """datetime; Accelerometer1RMS; Accelerometer2RMS; Current; Pressure; Temperature; Thermocouple; Voltage; Volume Flow RateRMS
    """ + '\n'.join(model.X)
    df=pd.read_csv(io.StringIO(model.X),sep=';', index_col='datatime', parse_dates=True)
    result = inference.model_inference(df, loaded_model, stsc, ucl)
    return{"result" : ''.join(map(str,result))}  

def main():
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000)

if __name__ =="__main__":
    main()