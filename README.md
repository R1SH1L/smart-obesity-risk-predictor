# Smart Obesity Risk Predictor and Health Advisor

This capstone project leverages machine learning models to classify BMI, predict body fat percentage, and provide health insights based on exercise and anthropometric data.

## Project Structure
```
smart_obesity_project/
├── data/                   # Input datasets
├── notebooks/              # Model development and training
├── models/saved_models/    # Trained models and scalers
├── app/                    # Example Streamlit demo (optional)
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
```

## Features
1. BMI Classification using logistic regression.
2. Body Fat Percentage Prediction using a random forest regressor.
3. Exercise Data Analysis for simple health recommendations.
4. Demo App (Optional):  
   A Streamlit app is included as an example interface for model demonstration.

## How to Use

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train or retrain models:**  
   Run the notebooks in `/notebooks` to train and save the models.

3. **Use the models in your own code:**  
   Load the models directly in Python:
   ```python
   import joblib
   bmi_model = joblib.load('models/saved_models/bmi_model.pkl')
   bodyfat_model = joblib.load('models/saved_models/bodyfat_model.pkl')
   bodyfat_scaler = joblib.load('models/saved_models/bodyfat_scaler.pkl')
   ```
   Prepare your input data as a pandas DataFrame with the same features and order as used during training.

4. **Run the Streamlit (demo app):**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   > **Note:** The Streamlit app is provided as a demonstration and is not required for using the models.

## Datasets
- `bmi.csv`: Gender, height, weight, and BMI index.
- `bodyfat.csv`: Age, weight, waist, and other measurements with body fat %.
- `exercise_dataset.csv`: Duration, calories, heart rate, temperature, etc.

## Models
- **BMI Classification:** Logistic Regression
- **Body Fat Prediction:** Random Forest Regressor
- **Exercise Recommendations:** Simple rule-based logic

## Disclaimer

These models provide estimates for educational purposes and should not be used as a substitute for professional medical advice..