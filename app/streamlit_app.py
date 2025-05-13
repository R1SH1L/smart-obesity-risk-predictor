import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Health Metrics Prediction",
    page_icon="💪",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    base_path = 'c:/Users/RISHIL T K/OneDrive/Desktop/DL - PROJECT/models/saved_models/'
    try:
        bmi_model = joblib.load(f'{base_path}bmi_model.pkl')
        bodyfat_model = joblib.load(f'{base_path}bodyfat_model.pkl')
        bodyfat_scaler = joblib.load(f'{base_path}bodyfat_scaler.pkl')
        return bmi_model, bodyfat_model, bodyfat_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

bmi_model, bodyfat_model, bodyfat_scaler = load_models()

# Title and introduction
st.title("🏃‍♂️ Health Metrics Prediction")
st.markdown("""
Choose a prediction type and enter your details to get started!
""")

# Create tabs for different predictions
tab1, tab2 = st.tabs(["BMI Prediction", "Body Fat Prediction"])

with tab1:
    st.markdown("### BMI Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        
        if st.button("Calculate BMI"):
            if bmi_model is not None:
                try:
                    # Convert gender to numeric (0 for Female, 1 for Male)
                    gender_numeric = 1 if gender == "Male" else 0
                    input_data = pd.DataFrame([[gender_numeric, height, weight]], 
                                           columns=['Gender', 'Height', 'Weight'])
                    prediction = bmi_model.predict(input_data)
                    
                    # Map prediction to BMI category
                    bmi_categories = {
                        0: "Underweight",
                        1: "Normal Weight",
                        2: "Overweight",
                        3: "Obese"
                    }
                    category = bmi_categories.get(prediction[0], "Unknown")
                    
                    st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px;'>
                        <h3 style='text-align: center; color: #0066cc;'>
                            BMI Category: {category}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("BMI model not loaded")

with tab2:
    st.markdown("### Body Fat Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        # Collect measurements in the exact order expected by the model
        density = 0  # Default value
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        weight = st.number_input("Weight (kg)", key="bf_weight", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", key="bf_height", min_value=100.0, max_value=250.0, value=170.0)
        neck = st.number_input("Neck Circumference (cm)", min_value=20.0, max_value=60.0, value=35.0)
        chest = st.number_input("Chest Circumference (cm)", min_value=60.0, max_value=150.0, value=90.0)
        abdomen = st.number_input("Abdomen Circumference (cm)", min_value=60.0, max_value=150.0, value=85.0)
        hip = st.number_input("Hip Circumference (cm)", min_value=60.0, max_value=150.0, value=95.0)
        thigh = st.number_input("Thigh Circumference (cm)", min_value=30.0, max_value=100.0, value=55.0)
        knee = st.number_input("Knee Circumference (cm)", min_value=20.0, max_value=60.0, value=35.0)
        ankle = st.number_input("Ankle Circumference (cm)", min_value=15.0, max_value=40.0, value=22.0)
        biceps = st.number_input("Biceps Circumference (cm)", min_value=20.0, max_value=50.0, value=30.0)
        forearm = st.number_input("Forearm Circumference (cm)", min_value=15.0, max_value=40.0, value=25.0)
        wrist = st.number_input("Wrist Circumference (cm)", min_value=10.0, max_value=30.0, value=17.0)
        
        if st.button("Calculate Body Fat"):
            if bodyfat_model is not None and bodyfat_scaler is not None:
                try:
                    # Create feature list in the exact order expected by the model
                    features = ['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 
                              'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 
                              'Forearm', 'Wrist']
                    
                    # Create input values in the same order as features
                    input_values = [density, age, weight, height, neck, chest, abdomen, 
                                  hip, thigh, knee, ankle, biceps, forearm, wrist]
                    
                    # Create DataFrame with ordered features
                    input_data = pd.DataFrame([input_values], columns=features)
                    
                    # Scale the input data
                    scaled_data = bodyfat_scaler.transform(input_data)
                    prediction = bodyfat_model.predict(scaled_data)
                    
                    st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px;'>
                        <h3 style='text-align: center; color: #0066cc;'>
                            Predicted Body Fat Percentage: {prediction[0]:.2f}%
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show interpretation of results
                    if prediction[0] < 6:
                        st.warning("⚠️ Body fat percentage is below essential fat levels")
                    elif prediction[0] < 14:
                        st.success("✅ Athletic range")
                    elif prediction[0] < 18:
                        st.success("✅ Fitness range")
                    elif prediction[0] < 25:
                        st.success("✅ Acceptable range")
                    else:
                        st.warning("⚠️ Body fat percentage is above recommended levels")
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("Body Fat models not loaded")

# Add explanatory notes
st.markdown("""
### 📝 Notes
- BMI Categories:
  - Underweight: < 18.5
  - Normal Weight: 18.5 - 24.9
  - Overweight: 25 - 29.9
  - Obese: ≥ 30
- Body Fat Percentage Guidelines:
  - Essential Fat (Men): 2-5%, (Women): 10-13%
  - Athletes (Men): 6-13%, (Women): 14-20%
  - Fitness (Men): 14-17%, (Women): 21-24%
  - Acceptable (Men): 18-24%, (Women): 25-31%
- These predictions are estimates and should not replace professional medical advice
""")