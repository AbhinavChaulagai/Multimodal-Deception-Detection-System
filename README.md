 Multimodal Deception Detection System

A full-stack AI-powered deception detection system that analyzes video, audio, and text transcription data to detect potential deceptive behavior using multimodal machine learning techniques.
 Overview

This project leverages multimodal data processing to improve deception detection accuracy by combining:

 Video Analysis – Facial expressions, micro-expressions, and behavioral cues

 Audio Analysis – Voice stress, tone, pitch variation, and speech patterns

 Text Transcription Analysis – Linguistic patterns, sentiment, and semantic inconsistencies

By integrating all three modalities, the system enhances prediction reliability compared to single-modality approaches.

 Architecture

Frontend:

Hosted on Netlify

Handles file uploads (video/audio) and text input

Displays prediction results with confidence scores

Backend:

Deployed on Render

REST API for processing multimodal inputs

ML inference pipeline for video, audio, and text models

Processing Pipeline:

User uploads video/audio or enters text

Server extracts features from:

Video frames

Audio signals

Transcribed speech

Each modality is analyzed independently

Multimodal fusion layer combines predictions

Final deception score is returned

 Key Features

Multimodal ML-based detection

Real-time server inference

Modular model design (independent pipelines for each modality)

Scalable cloud deployment

Clean frontend interface

 Tech Stack

Frontend

JavaScript / React (if applicable)

Netlify Deployment

Backend

Python (FastAPI / Flask – adjust if needed)

Render Deployment

Machine Learning

OpenCV (video processing)

Librosa (audio processing)

NLP models (for transcription analysis)

Multimodal fusion model

 Installation (Local Setup)
# Clone the repository
git clone https://github.com/AbhinavChaulagai/Multimodal-Deception-Detection-System.git

# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup
cd frontend
npm install
npm start
 Deployment

Backend deployed on Render

Frontend deployed on Netlify

 Use Cases

Behavioral research

Interview screening analysis

Forensic linguistics research

Academic experimentation with multimodal AI

⚠️ Disclaimer

This system is designed for research and experimental purposes only.
Deception detection using AI is probabilistic and should not be used as a sole basis for decision-making in legal, hiring, or high-stakes environments.
