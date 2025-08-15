# HRRS Land Cover Classification

🌍 **Land Cover Classification using HRRS (High-Resolution Remote Sensing) Satellite Images**  

This project provides an **AI-powered tool** to classify land cover types from high-resolution HRRS satellite images. The model is trained on the **EuroSAT RGB dataset** and can identify 10 different land cover classes with high accuracy.

---

## 📁 Dataset

**EuroSAT RGB Dataset:**  
The dataset consists of satellite images covering various land cover types.  

[Download the dataset here](https://drive.google.com/drive/folders/1vo7I6OvK_tLv_L9CGFtEUfc0mB1iQIvb?usp=sharing)

**Classes in the dataset:**

| Class Index | Class Label                 |
|------------:|----------------------------|
| 0           | AnnualCrop                 |
| 1           | Forest                     |
| 2           | HerbaceousVegetation       |
| 3           | Highway                    |
| 4           | Industrial                 |
| 5           | Pasture                    |
| 6           | PermanentCrop              |
| 7           | Residential                |
| 8           | River                      |
| 9           | SeaLake                    |

---

## 🧠 Model Architecture

The classification model is based on **ResNet-50**, utilizing **Identity Blocks** and **Convolutional Blocks** to extract robust features from satellite images.  

- Achieved **>84% validation accuracy** on EuroSAT RGB dataset.  
- Input image size: 64×64 pixels (RGB).  
- Output: 10 land cover classes.

---

## ⚡ Live Demo / Streamlit App

A user-friendly **Streamlit web application** is provided to upload satellite images and get instant land cover predictions.

**Model Download Link:**  
[Download Trained Model](https://drive.google.com/drive/folders/1Hp2pF1OtUazdGGRM9cywQbJYLmDLAWs4?usp=drive_link)

**Steps to run the app locally:**

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

2. pip install -r requirements.txt

3. LOCAL_MODEL_PATH = r"C:\Users\YOUR_USER\PATH_TO_MODEL\lulc_2_epoch" (UPDATE IF NEEDED AS YOU)

4. streamlit run app.py

