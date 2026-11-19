# Plant Disease Detection using CNN
## AI-Powered Plant Leaf Disease Detection and Recommendation System

**JNTUA B.Tech CSE Final Year Project — 2024-25**
**CVRT Engineering College, Tadipatri**
**Student:** Chaganti Nithin — HT NO: 239B1A0528

---

## 🌿 Project Overview

This system uses a fine-tuned **MobileNetV2 CNN** to detect and classify **38 plant diseases** across **14 crop species** from leaf images. It achieves **92.4% accuracy** on the PlantVillage dataset (87,000+ images).

---

## 📁 Project Structure

```
plant-disease-detection/
├── frontend/
│   ├── index.html          ← Landing page
│   ├── predict.html        ← Disease prediction (upload & analyze)
│   ├── results.html        ← Detailed results & treatment
│   ├── about.html          ← About the project
│   ├── dashboard.html      ← Prediction history & stats
│   ├── css/
│   │   └── style.css       ← Complete stylesheet
│   └── js/
│       └── main.js         ← All frontend logic
├── backend/
│   ├── app.py              ← Main Flask application
│   ├── model.py            ← ML model prediction wrapper
│   ├── routes.py           ← All API endpoints
│   ├── database.py         ← SQLite database operations
│   └── utils.py            ← Helper functions
├── ml/
│   ├── train.py            ← CNN model training (MobileNetV2)
│   ├── preprocess.py       ← Data cleaning + augmentation
│   ├── evaluate.py         ← Model evaluation + metrics
│   ├── plant_disease_model.h5 ← Saved trained CNN model
│   ├── class_labels.json   ← 38 disease class index mappings
│   ├── evaluation_metrics.json ← Evaluation benchmark scores
│   ├── confusion_matrix.png ← Confusion matrix plot
│   └── per_class_accuracy.png ← Per-class accuracy plot
├── sample_images/          ← 5 demo leaf images for live viva testing
├── datasets/
│   └── README.md           ← Dataset download instructions
├── tests/
│   ├── test_model.py       ← ML model unit tests
│   ├── test_api.py         ← API endpoint tests
│   └── test_results.txt    ← Sample test outputs
├── requirements.txt        ← All dependencies
├── setup.py                ← One-time setup script
├── run.py                  ← Start the entire application
└── README.md               ← This file
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended for model training)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Setup Database
```bash
python setup.py
```

### Step 3: Train the Model (first time only)
```bash
python ml/train.py
```
> ⚠️ First-time training takes 10-15 minutes. The trained model is saved to `ml/plant_disease_model.h5`. Alternatively, download pre-trained weights from the dataset README.

### Step 4: Run the Application
```bash
python run.py
```

### Step 5: Open in Browser
```
http://localhost:5000
```

---

## 📊 Dataset

- **Name:** PlantVillage Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/emmarex/plantdisease
- **Size:** 87,000+ leaf images
- **Classes:** 38 (14 crop species × disease variants + healthy)
- **Format:** RGB images (256×256 pixels)

**Supported Crops:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|---|---|---|
| **Frontend** | HTML5 + CSS3 + JavaScript | ES6+ |
| **Backend** | Python Flask | 2.3.3 |
| **ML Framework** | TensorFlow + Keras | 2.13.0 |
| **CNN Architecture** | MobileNetV2 (Transfer Learning) | - |
| **Database** | SQLite3 | 3.x |
| **Image Processing** | OpenCV + Pillow | 4.8.1 / 10.0.1 |
| **ML Utilities** | scikit-learn | 1.3.0 |
| **API** | Flask-RESTful + Flask-CORS | 0.3.10 / 4.0.0 |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **Accuracy** | 92.4% |
| **Precision** | 91.8% |
| **Recall** | 93.1% |
| **F1-Score** | 92.4% |
| **Training Dataset** | 69,600 images (80%) |
| **Test Dataset** | 17,400 images (20%) |

---

## 🌾 Disease Classes Supported

The system detects 38 plant disease conditions. Examples:
- Apple Scab, Apple Black Rot, Apple Cedar Rust
- Corn Gray Leaf Spot, Corn Common Rust, Corn Northern Blight
- Tomato Bacterial Spot, Tomato Early Blight, Tomato Late Blight, Tomato Mosaic Virus
- Potato Early Blight, Potato Late Blight
- Grape Black Rot, Grape Leaf Blight
- ... and 20+ more

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Upload leaf image → get disease prediction |
| `GET` | `/api/history` | Get prediction history |
| `GET` | `/api/diseases` | Get all supported disease classes |
| `POST` | `/api/register` | User registration |
| `POST` | `/api/login` | User login |
| `GET` | `/api/stats` | System statistics |

---

## 👥 Team

| Role | Name | HT Number |
|---|---|---|
| Team Leader — Frontend + Integration | Chaganti Nithin | 239B1A0528 |
| Backend + API + Database | Team Member 2 | TBD |
| ML Model + Training + Testing | Team Member 3 | TBD |
| Documentation + Testing | Team Member 4 | TBD |

**Guide:** To be assigned by HOD  
**Department:** CSE, CVRT Engineering College, Tadipatri  
**University:** JNTUA | Academic Year: 2024-25

---

## 📝 License

This project is developed for academic purposes under JNTUA B.Tech CSE curriculum (R20 Regulation). Not for commercial use.
