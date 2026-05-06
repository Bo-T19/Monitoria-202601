from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle

# Cargar el modelo entrenado
with open('modelo_spam.pkl', 'rb') as f:
    modelo_spam = pickle.load(f)

app = FastAPI(
    title="API de Clasificación de Spam",
    description="Una API para clasificar mensajes (en inglés) como spam o no spam utilizando un modelo tipo Random Forest entrenado con TF-IDF.",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir el modelo de datos para la solicitud
class Message(BaseModel):
    text: str = Field(..., min_length=10, example="Congratulations! You've won a free ticket to the Bahamas. Click here to claim your prize.")


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Spam. Use el endpoint /predict para clasificar mensajes."}

@app.post("/predict")
def predict_spam(message: Message):
    """
    Endpoint para predecir si un mensaje es spam o no spam.
    
    - **text**: El mensaje a clasificar (en inglés).
    
    Retorna un diccionario con la predicción ("spam" o "ham").
    """
    prediccion = modelo_spam.predict([message.text])[0]
    probabilidad = modelo_spam.predict_proba([message.text])[0]
    class_labels = modelo_spam.classes_.tolist()
    spam_index = class_labels.index("spam")

    return { "label": prediccion ,
            "is_spam": prediccion == "spam" ,
            "spam_probability": round(probabilidad[spam_index], 4) 
    }   


