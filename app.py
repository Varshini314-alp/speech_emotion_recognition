
import os
import tempfile

import keras
import librosa
import numpy as np
import joblib

from deepface import DeepFace

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(title="Speech and Facial Emotion Recognition")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


# --------------------------------------------------
# LOAD AUDIO EMOTION MODEL
# --------------------------------------------------

model = None
encoder = None

try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("Audio emotion model loaded successfully.")
    else:
        print("Audio model file not found:", MODEL_PATH)

except Exception as e:
    print("Audio model loading error:", repr(e))


# --------------------------------------------------
# LOAD LABEL ENCODER
# --------------------------------------------------

try:
    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(ENCODER_PATH)
        print("Emotion encoder loaded successfully.")
    else:
        print("Encoder file not found:", ENCODER_PATH)

except Exception as e:
    print("Encoder loading error:", repr(e))


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# --------------------------------------------------
# CAMERA PAGE
# --------------------------------------------------

@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="camera.html",
        context={}
    )


# --------------------------------------------------
# AUDIO EMOTION DETECTION
# --------------------------------------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict_audio(
    request: Request,
    file: UploadFile = File(...)
):

    if model is None or encoder is None:
        return HTMLResponse(
            "Audio model or encoder did not load. "
            "Check the terminal.",
            status_code=500
        )

    temp_path = None

    try:

        # Save uploaded audio temporarily
        suffix = os.path.splitext(
            file.filename or "audio.wav"
        )[1] or ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Load audio
        audio, sr = librosa.load(
            temp_path,
            sr=22050,
            mono=True,
            duration=3,
            offset=0.5
        )

        if len(audio) == 0:
            raise ValueError("Uploaded audio is empty.")

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        features = np.mean(
            mfcc.T,
            axis=0
        ).reshape(1, -1)

        # Predict emotion
        probabilities = model.predict(
            features,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        emotion = str(
            encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        confidence = float(
            probabilities[predicted_index] * 100
        )

        # Display result page
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "method": "Audio Emotion Detection"
            }
        )

    except Exception as e:

        return HTMLResponse(
            f"Audio prediction error: {str(e)}",
            status_code=500
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# FACIAL EMOTION DETECTION
# --------------------------------------------------

@app.post("/facial-emotion")
async def facial_emotion(
    file: UploadFile = File(...)
):

    temp_path = None

    try:

        # Save uploaded camera image temporarily
        suffix = os.path.splitext(
            file.filename or "image.jpg"
        )[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Detect facial emotions
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=["emotion"],
            enforce_detection=True
        )

        # Handle DeepFace result
        if isinstance(result, list):

            if not result:
                raise ValueError("No face detected.")

            result = result[0]

        # Get dominant emotion
        emotion = result["dominant_emotion"]

        confidence = float(
            result["emotion"][emotion]
        )

        # Return JSON response to camera.html
        return JSONResponse(
            content={
                "emotion": str(emotion),
                "confidence": round(confidence, 2),
                "method": "Facial Emotion Detection"
            }
        )

    except Exception as e:

        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "Application running",
        "audio_model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "facial_detection": "Available"
    }
