# Speech and Facial Emotion Recognition 🎙️📷

An AI-powered web application that identifies human emotions through two input methods: **voice recordings and facial images**. The project combines Machine Learning and Deep Learning techniques to analyze audio signals and facial expressions.

Built using Python, FastAPI, TensorFlow, Keras, Librosa, and DeepFace, the application provides a simple interface for emotion prediction.

## 🚀 Project Highlights

- Dual-mode emotion recognition using audio and facial images.
- Audio feature extraction using MFCC.
- Deep learning-based audio emotion classification.
- Facial emotion analysis using DeepFace.
- Web-based application developed with FastAPI.
- Prediction results displayed with confidence scores.
- Separate interfaces for audio and camera detection.
- API endpoints for connecting the frontend and backend.
- Temporary file handling for uploaded audio and images.
- Modular backend structure for easier maintenance and future improvements.

## ✨ Key Features

### 1. Audio Emotion Detection 🎙️

- Upload an audio recording through the web interface.
- Process audio files using Librosa.
- Convert audio to mono and resample it to 22,050 Hz.
- Extract 40 MFCC (Mel-Frequency Cepstral Coefficients) features.
- Calculate mean MFCC values to prepare model input.
- Use a trained Keras model to predict the emotion.
- Decode the predicted class using the saved label encoder.
- Display the predicted emotion and confidence score.

### 2. Facial Emotion Detection 📷

- Open the camera page through the web application.
- Capture or upload a facial image.
- Process the image using DeepFace.
- Analyze facial expressions for emotion classification.
- Identify the dominant detected emotion.
- Display the emotion and its confidence score.
- Handle cases where a face cannot be detected.

### 3. Web Application Features 🌐

- FastAPI-based backend.
- HTML templates for user interaction.
- Separate audio and camera pages.
- Prediction result page with emotion information.
- JSON response for facial emotion detection.
- API documentation through Swagger UI.
- Health-check endpoint to verify application status.

## 🛠️ Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep learning framework |
| Keras | Loading and running the trained audio model |
| Librosa | Audio processing and feature extraction |
| NumPy | Numerical calculations |
| Joblib | Loading the label encoder |
| DeepFace | Facial emotion analysis |
| OpenCV | Image processing support |
| FastAPI | Backend development |
| HTML & CSS | Frontend design |
| JavaScript | Browser camera interaction |
| Uvicorn | Running the FastAPI server |
| VS Code | Development environment |
| Git & GitHub | Version control and project hosting |

## 🧠 Project Workflow

### Audio Emotion Recognition

1. User uploads an audio recording.
2. The application reads and preprocesses the audio.
3. Librosa extracts MFCC features.
4. Mean feature values are calculated.
5. The trained Keras model predicts the emotion class.
6. The label encoder converts the predicted class into an emotion name.
7. The application displays the prediction and confidence score.

### Facial Emotion Recognition

1. User opens the camera interface.
2. A facial image is captured or uploaded.
3. The image is temporarily stored for processing.
4. DeepFace analyzes the facial expression.
5. The dominant emotion is identified.
6. The application returns the emotion and confidence score.

## 📊 Audio Feature Extraction

The audio prediction process uses MFCC (Mel-Frequency Cepstral Coefficients).

**MFCC feature extraction helps represent important characteristics of audio signals.**

The project uses:
- Sampling rate: 22,050 Hz
- Audio channels: Mono
- Audio duration: Up to 3 seconds
- Number of MFCC coefficients: 40
- Feature aggregation: Mean of MFCC values

These extracted features are passed to the trained audio classification model.

## 📂 Project Structure

```text
speech_emotion_recognition/
│
├── app.py
├── requirements.txt
├── emotion_model.keras
├── label_encoder.pkl
├── README.md
├── .gitignore
│
└── templates/
    ├── index.html
    ├── camera.html
    └── result.html
```

**Important:** The audio model and encoder files must be available in the project directory for audio prediction to work. The virtual environment is not included in the repository.

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Varshini314-alp/speech_emotion_recognition.git
```

### Step 2: Open the Project Folder

```bash
cd speech_emotion_recognition
```

### Step 3: Create a Virtual Environment

Python 3.12 is recommended for this project.

```bash
py -3.12 -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
python -m uvicorn app:app --reload
```

## 🌐 Access the Application

After starting the server, open these URLs in your browser:

| Page | URL |
|---|---|
| Home / Audio Detection | http://127.0.0.1:8000/ |
| Camera Detection | http://127.0.0.1:8000/camera |
| API Documentation | http://127.0.0.1:8000/docs |
| Health Check | http://127.0.0.1:8000/health |

Allow browser camera permission when using the webcam feature.

## 🔗 API Endpoints

| HTTP Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays the home page |
| GET | `/camera` | Opens the camera page |
| POST | `/predict` | Predicts emotion from uploaded audio |
| POST | `/facial-emotion` | Detects facial emotion from an uploaded image |
| GET | `/health` | Checks application status and model loading |

## 💡 Implementation Highlights

- **Audio preprocessing:** Standardizes audio input before feature extraction.
- **Feature engineering:** Uses MFCC coefficients to represent audio characteristics.
- **Model inference:** Loads a previously trained Keras model for audio prediction.
- **Label decoding:** Uses Joblib to load the saved label encoder.
- **Facial analysis:** Integrates DeepFace for facial emotion detection.
- **Backend integration:** Uses FastAPI routes to connect the prediction logic with the frontend.
- **Error handling:** Handles missing model files, empty audio, image-processing errors, and face-detection failures.
- **Temporary file management:** Removes temporary uploaded files after processing.

## 🎯 Project Objectives

- Develop an AI-based system for recognizing emotions from multiple input types.
- Apply audio preprocessing and feature extraction techniques.
- Understand how deep learning models perform classification.
- Integrate facial emotion analysis into a web application.
- Build and connect backend APIs with frontend templates.
- Gain practical experience in deploying AI/ML models through a web framework.

## 🔮 Future Enhancements

- Real-time emotion detection from live audio.
- Continuous facial emotion monitoring through a webcam.
- Combined audio and facial emotion analysis.
- Improved model performance through additional training data.
- Emotion history and visualization dashboard.
- Support for additional audio formats and input conditions.
- Cloud deployment for easier access.
- Improved user interface and accessibility.

## ⚠️ Limitations

- Audio prediction depends on the quality and duration of the uploaded recording.
- Facial emotion detection requires a clearly detectable face.
- Background noise, lighting, image quality, and recording conditions may affect predictions.
- Confidence scores represent model predictions and should not be treated as proof of a person's actual emotional state.

## 👩‍💻 Developer

**Varshini M.**

AI/ML Developer | BCA Graduate

GitHub: [Varshini314-alp](https://github.com/Varshini314-alp)

Project Repository: [Speech and Facial Emotion Recognition](https://github.com/Varshini314-alp/speech_emotion_recognition)

---

⭐ If you find this project interesting, feel free to explore the repository!

**Technologies:** Python | Machine Learning | Deep Learning | TensorFlow | Keras | Librosa | DeepFace | FastAPI
