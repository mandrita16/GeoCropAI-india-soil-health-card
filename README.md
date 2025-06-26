# GeoCropAI-india-soil-health-card
GeoCropAI : India Soil Health Card Analysis System

A machine learning–powered system to analyze soil health, recommend fertilizers and suggest crop suitability based on Indian agro-climatic conditions.

# 📌 Project Overview
AgriIntel leverages synthetic data modeled on the India Soil Health Card Dataset to deliver:

✅ Soil Health Classification (Excellent to Critical)

✅ Fertilizer Recommendations

✅ Suitable Crop & Vegetable Suggestions

✅ Model Accuracy: Up to 96.25%


# Dataset

🔗 Original Source:

India Soil Health Card - Google Research Dataset
```bash
https://github.com/google-research-datasets/india-soil-health-card
```

📄 Note: Since the original dataset is stored in Docker format, a cleaned and simulated CSV-based version has been generated for direct use.

# 🚀 Installation & Setup

✅ Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/GeoCropAI-india-soil-health-card.git
cd GeoCropAI-india-soil-health-card
```

✅ Step 2: Create and Activate Virtual Environment
```bash

# Create environment (Windows/Mac/Linux)
python -m venv crop_env
```
# Activate environment
# For Windows:
```bash
crop_env\Scripts\activate

# For Mac/Linux:
source crop_env/bin/activate
```
✅ Step 3: Install Required Libraries
```bash

pip install -r requirements.txt
```
If requirements.txt is not present, install manually:

```bash

pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask joblib
```
# 📂 Project Structure


    
# 🔍 How to Use
⚙️ 1. Run the Prediction Engine
```bash

python soil_health_analysis.py
```


# 💡 Features
It covers all major agro-climatic regions, including but not limited to:

```bash

Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Delhi,
Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala,
Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland,
Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura,
Uttar Pradesh, Uttarakhand, West Bengal, Jammu & Kashmir, Ladakh, etc.
```
✅ Tested on soil-health patterns across all 28 states + UTs

Fertilizer dosage calculation based on nutrient deficiency

Crop compatibility matrix based on pH, temperature, and rainfall



👨‍🏫 Mentor-Ready Justification for Dataset
Since the official dataset is in Docker format and hard to parse in short hackathon durations, we built a simulated CSV-based dataset mirroring the structure and value ranges of the original. This lets us test our ML models reliably while adhering to the constraints of offline judging and limited time.

# 📈 Model Performance
Model	Accuracy
XGBoost	95.80%
Random Forest	94.65%
Ensemble	96.25% ✅

# 🛠️ Future Scope
*Mobile integration using Flutter

*Hindi & all regional language support

* # 🛠️ Implementation Notes
        In the future, Flask will be used to build app.py, which will expose prediction APIs and integrate the ML model with the frontend.
        
        The trained model (xgb_model.pkl) and feature scaler (scaler.pkl) were generated using the following commands inside soil_health_analysis.py:
        
       
        import joblib
        
        # Save model and scaler
        joblib.dump(best_model, 'xgb_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        These files will be loaded inside app.py during runtime to serve predictions.
        
        
*API exposure for government portal integration

# 🙏 Acknowledgments
Ministry of Agriculture, India

Google Research Datasets



