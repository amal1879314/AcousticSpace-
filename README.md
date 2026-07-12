# 🎙️ AcousticSpace

**Physics-Based Deepfake Audio Detection using Room Impulse Response (RIR) and Transformer Models**

AcousticSpace is an AI-powered audio forensics platform that detects deepfake audio by analyzing **Room Impulse Response (RIR)**, environmental acoustics, and breathing patterns instead of relying only on voice characteristics.

Unlike traditional deepfake detectors that focus on vocal artifacts, AcousticSpace examines the **physical properties of sound**, enabling it to identify inconsistencies between a speaker's voice and the surrounding acoustic environment.

---

## ✨ Features

- 🎵 Audio file upload and analysis
- 🏠 Room Impulse Response (RIR) extraction
- 🌊 Audio waveform and spectrogram visualization
- 🌬️ Breathing pattern analysis
- 🤖 Transformer-based deepfake detection
- 📊 Confidence score prediction
- ⚡ Real-time inference with FastAPI
- 💻 Interactive React dashboard

---

## 🛠️ Tech Stack

### Backend
- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- Librosa
- NumPy
- SciPy

### Frontend
- React
- TypeScript
- Tailwind CSS
- Wavesurfer.js

### DevOps
- Docker
- GitHub Actions

---

## 🧠 Detection Pipeline

1. Upload an audio file.
2. Preprocess and extract acoustic features using Librosa.
3. Analyze Room Impulse Response (RIR), reverberation, and breathing patterns.
4. Perform inference using a fine-tuned Audio Spectrogram Transformer (AST).
5. Display the prediction, confidence score, and suspicious audio segments.

---

## 📊 Dataset

The model is trained and evaluated using publicly available speech datasets such as:

- ASVspoof
- LibriSpeech
- VoxCeleb

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/yourusername/AcousticSpace.git
cd AcousticSpace
```

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🎯 Project Goal

Build a reliable deepfake audio detection system that leverages **room acoustics and environmental physics** to identify synthetic speech with greater robustness than traditional voice-based detection methods.

---

> **Status:** 🚧 Currently under development.
