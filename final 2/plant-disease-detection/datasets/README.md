# Dataset Download Instructions

## PlantVillage Dataset

**Dataset Name:** PlantVillage Dataset  
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/emmarex/plantdisease  
**Size:** ~2.6 GB (87,000+ images)  
**Classes:** 38 plant disease categories

---

## How to Download

### Method 1: Kaggle CLI (Recommended)
```bash
# Install kaggle
pip install kaggle

# Place your kaggle.json API key in ~/.kaggle/
# Get it from: https://www.kaggle.com/account → Create API Token

kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d datasets/plantvillage
```

### Method 2: Manual Download
1. Go to https://www.kaggle.com/datasets/emmarex/plantdisease
2. Sign in with your Kaggle account (free)
3. Click "Download" button
4. Extract the ZIP file to `datasets/plantvillage/`

---

## Expected Directory Structure After Download

```
datasets/
├── README.md              ← This file
└── plantvillage/
    ├── Apple___Apple_scab/
    │   ├── image001.JPG
    │   ├── image002.JPG
    │   └── ...
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
    ├── Corn_(maize)___Common_rust_/
    ├── Corn_(maize)___Northern_Leaf_Blight/
    ├── Corn_(maize)___healthy/
    ├── Grape___Black_rot/
    ├── Grape___Esca_(Black_Measles)/
    ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
    ├── Grape___healthy/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    ├── Potato___healthy/
    ├── Tomato___Bacterial_spot/
    ├── Tomato___Early_blight/
    ├── Tomato___Late_blight/
    ├── Tomato___Leaf_Mold/
    ├── Tomato___Septoria_leaf_spot/
    ├── Tomato___Spider_mites Two-spotted_spider_mite/
    ├── Tomato___Target_Spot/
    ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    ├── Tomato___Tomato_mosaic_virus/
    ├── Tomato___healthy/
    └── ... (38 total classes)
```

---

## Dataset Statistics

| Property | Value |
|---|---|
| Total Images | 87,000+ |
| Number of Classes | 38 |
| Number of Crop Species | 14 |
| Image Format | JPG/JPEG |
| Image Size | 256×256 pixels |
| Color Mode | RGB |
| Source | Penn State University (2016) |
| License | CC0: Public Domain |

---

## 14 Supported Crop Species

1. Apple
2. Blueberry
3. Cherry
4. Corn (Maize)
5. Grape
6. Orange
7. Peach
8. Bell Pepper
9. Potato
10. Raspberry
11. Soybean
12. Squash
13. Strawberry
14. Tomato

---

## Note

If you cannot download the full dataset (2.6 GB), use the smaller split version:
- **URL:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- This contains a pre-split train/validation set (same images)
- Place under `datasets/plantvillage/train/` and `datasets/plantvillage/val/`
