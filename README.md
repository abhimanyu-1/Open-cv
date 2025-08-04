# 🔍 VisionX: A Multifunctional Image Understanding Toolkit

Welcome to **VisionX** — a complete image intelligence pipeline built with deep learning and traditional image processing. This project brings together powerful modules like image captioning, line detection/removal, text-to-image synthesis, and image-to-text understanding, all seamlessly integrated with a backend-ready format for scalable deployment and experimentation.

---

## 🚀 Features

### 🖼️ Image Captioning
- **Model**: [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xl) powered caption generation.
- **Goal**: Automatically generate rich, descriptive captions for input images.

### 📏 Line Detection & Removal
- Detect horizontal and vertical lines using OpenCV and morphological operations.
- Remove unwanted lines (e.g., form borders, table outlines) while preserving text or object info.

### 🔁 Text ↔ Image Pipelines
- **Image-to-Text**: Extract captions or structured descriptions from images.
- **Text-to-Image**: Use diffusion models (e.g., `stabilityai/stable-diffusion`) to synthesize images from descriptive prompts.

### 🧠 Visual Understanding
- Visual Question Answering (VQA) using vision-language models.
- Damage or defect detection from vehicle images using FLAN-T5 or Mistral-based reasoning.

### 📦 Vector Store Integration
- All image captions and descriptions are embedded and stored in a **vector database** using `SentenceTransformers` + `FAISS`, enabling fast image/text similarity search and retrieval.

---

## 📂 Folder Structure

