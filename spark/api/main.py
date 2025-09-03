from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# (import model) from xx import load_model
# (import preprocessed function)from xx import preprocess_features

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/predict')
def predict():
    return {'your health condition': 'You are very healthy my friend!'}


# app.state.model = load_model()

# @app.get("/predict")
# def predict(
#         #placeholder columns for user imput, examples
#         # pickup_datetime: str,  # 2014-07-06 19:18:00
#         #pickup_longitude: float,
#     ):      
    

    
#     # locals() gets us all of our arguments back as a dictionary
#     # https://docs.python.org/3/library/functions.html#locals
#     X_pred = pd.DataFrame(locals(), index=[0])

   

    
#     assert model is not None

#     X_processed = preprocess_features(X_pred)
#     y_pred = model.predict(X_processed)
#     return dict(prediction=float(y_pred))

#     # ⚠️ fastapi only accepts simple Python data types as a return value
#     # among them dict, list, str, int, float, bool
#     # in order to be able to convert the api response to JSON
#     




@app.get("/")
def root():
    return dict(greeting="Hello! Welcome and wish you good health ❤️")