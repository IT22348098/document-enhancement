# DocEnhance AI — Medical Report Image Enhancement

A full-stack **Medical Report Image Enhancement System** with a **React frontend** and **FastAPI (Python) backend** that uses a trained U-Net model (`.keras` file) for document image enhancement (deblurring + denoising).

## 🗂 Repository Structure

```
frontend/                    # React app (Vite + React)
├── package.json
├── vite.config.js
├── index.html
└── src/
    ├── main.jsx
    ├── App.jsx / App.css
    ├── index.css
    ├── components/
    │   ├── Header.jsx / Header.css
    │   ├── UploadZone.jsx / UploadZone.css
    │   ├── ImageGallery.jsx / ImageGallery.css
    │   ├── ComparisonSlider.jsx / ComparisonSlider.css
    │   ├── Lightbox.jsx / Lightbox.css
    │   └── MetricsCard.jsx / MetricsCard.css
    └── utils/
        └── api.js

backend/
├── main.py                  # FastAPI application
├── enhance.py               # Model loading + patch-based inference
├── utils.py                 # Helper functions
├── requirements.txt
└── model/
    └── .gitkeep             # Place best_model.keras here
```

---

## 🚀 Quick Start

### 1. Train the Model (Google Colab)
- Open `document_enhancement.ipynb` in Google Colab
- Set the Runtime to **T4 GPU** → **Run All**
- The trained model is saved to Google Drive as `Medical_Report_checkpoints/best_model.keras`
- Expected training time: ~30–45 minutes

### 2. Backend Setup

```bash
cd backend

# Copy your trained model
cp /path/to/Medical_Report_checkpoints/best_model.keras model/

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### 4. Use It!

1. Open `http://localhost:5173`
2. Drag & drop medical report images (JPG, PNG, TIFF, BMP)
3. Click **"✨ Enhance All"**
4. Drag the comparison slider left/right to see before/after results
5. Click any image to open fullscreen (Lightbox) view
6. Download individual or all enhanced images

---

## ✨ Features

- **Drag-and-drop upload** with multi-file support
- **Before/After comparison slider** with mouse & touch support
- **Fullscreen Lightbox** — ESC to close, arrow keys to navigate
- **Quality metrics**: PSNR & SSIM (when ground truth is available)
- **Batch enhancement** — enhance all images in one request
- **Dark glassmorphism UI** — purple/navy accent theme
- **Responsive design** — mobile, tablet, and desktop

## 🧠 Model Architecture

The backend uses a **U-Net** trained on paired medical report images:
- **Input**: Blurry/noisy grayscale medical report image
- **Output**: Clean, deblurred, denoised version
- **Inference**: Patch-based (128×128 patches, 50% overlap) for arbitrary-size images

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check, model status |
| `POST` | `/api/enhance` | Enhance a single image |
| `POST` | `/api/enhance/batch` | Enhance multiple images |
