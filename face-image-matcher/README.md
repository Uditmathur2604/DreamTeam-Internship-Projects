# 👥 Face Image Matcher Using InsightFace + FAISS

> Developed as part of an internship project at **Dreamteam Technologies**

## 🔍 Overview

This project enables fast and accurate **face matching** from a dataset of face images using a webcam snapshot as input. It uses **InsightFace** for face detection and embedding, and **FAISS** for fast similarity search.

---

## 🧠 Core Features

- Extracts facial embeddings from a dataset of images
- Matches webcam-captured face with all known faces
- Displays top matches and saves them to a results folder
- Built for real-time applications with high accuracy

---

## 📁 Folder Structure

face-image-matcher/
├── event_photos/ # 🔒 [This folder should contain your private face images. DO NOT upload to GitHub.]
├── encode_faces.py # Extract and save face embeddings
├── face_matcher.py # Search similar faces using webcam
├── requirements.txt # All dependencies
└── README.md # Project documentation

**⚠️ Note:**  
The `event_photos/` folder is meant to be created locally and should include your own as well as others images. This folder is intentionally excluded from the GitHub repository for privacy and security.

---

## 🧰 Tech Stack

- Python
- OpenCV
- InsightFace (Buffalo_L model)
- FAISS
- NumPy, tqdm, scikit-learn

---

## ⚙️ How It Works

1. **Run `encode_faces.py`**  
   - Scans all images in `event_photos/`
   - Extracts embeddings
   - Saves `embeddings.npy` and `filenames.pkl`

2. **Run `face_matcher.py`**
   - Captures webcam photo
   - Detects face
   - Compares with dataset embeddings using FAISS
   - Shows matched images & saves to `matched_faces/`

---

## ▶️ Run Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Encode all dataset faces
python encode_faces.py

# 3. Start the matcher
python face_matcher.py
