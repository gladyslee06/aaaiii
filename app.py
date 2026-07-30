import streamlit as st
import pandas as pd
import pickle
import random
import os
import base64
import requests
import numpy as np

# ==================== Page configuration ====================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Debug toggle (off by default) ====================
debug_mode = st.sidebar.checkbox("Show debug info", value=False)

if debug_mode:
    st.sidebar.header("Debug Information")
    st.sidebar.write(f"**Current Directory:** `{os.getcwd()}`")
    model_files = ['diabetes_model.pkl', 'scaler.pkl', 'imputer.pkl']
    st.sidebar.write("**Model Files:**")
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            st.sidebar.write(f"  `{file}` ({size:.1f} KB)")
        else:
            st.sidebar.write(f"  `{file}` NOT FOUND")
    all_pkl = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if all_pkl:
        st.sidebar.write(f"**All PKL files:** {', '.join(all_pkl)}")

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .title-text {
        font-size: 3rem; font-weight: 700;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem; padding: 1rem;
    }
    .subtitle-text { text-align: center; color: #1a237e; font-size: 1.1rem; margin-bottom: 2rem; }
    .stNumberInput > div > div > input {
        border-radius: 10px; border: 2px solid #e0e0e0; padding: 0.5rem;
        transition: all 0.3s ease; height: 38px !important; min-height: 38px !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #1a237e; box-shadow: 0 0 0 3px rgba(26, 35, 126, 0.1);
    }
    .stNumberInput > div { margin-bottom: 0 !important; }
    .stNumberInput > div > div { height: 38px !important; min-height: 38px !important; }
    .validation-message {
        min-height: 24px; height: 24px; margin-top: 2px; margin-bottom: 8px;
        font-size: 0.8rem; display: flex; align-items: center; padding: 0 8px;
        border-radius: 6px; transition: all 0.2s ease;
    }
    .validation-message-empty { min-height: 24px; height: 24px; margin-top: 2px; margin-bottom: 8px; visibility: hidden; }
    .validation-error { color: #f5576c; background-color: #fff0f0; border-left: 3px solid #f5576c; }
    .validation-success { color: #28a745; background-color: #f0fff0; border-left: 3px solid #28a745; }
    .validation-warning { color: #856404; background-color: #fff3cd; border-left: 3px solid #ffc107; }
    .stNumberInput > label { margin-bottom: 0.2rem !important; font-weight: 500; }
    .stButton > button {
        width: 100%; background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        color: white; font-weight: 600; font-size: 1.2rem; padding: 0.75rem 2rem;
        border-radius: 50px; border: none; transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 35, 126, 0.4);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(26, 35, 126, 0.6); }
    .stButton > button:active { transform: translateY(0); }
    .bmi-underweight, .bmi-overweight {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: #1a237e;
    }
    .bmi-normal {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
        padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: #1a237e;
    }
    .bmi-obese {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
        padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: white;
    }
    .bmi-value { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; }
    .bmi-category { font-size: 1.5rem; font-weight: 600; margin: 0; }
    .bmi-description { font-size: 1rem; color: rgba(0,0,0,0.7); margin-top: 0.5rem; }
    .bmi-obese .bmi-description { color: rgba(255,255,255,0.8); }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; background: transparent; padding: 0; border-bottom: none !important; }
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem; background: transparent !important; border: none !important;
        border-radius: 0; color: #555; font-weight: 500; padding: 0.5rem 1.5rem; box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background: transparent !important; color: #1a237e; }
    .stTabs [aria-selected="true"] {
        background: transparent !important; color: #1565C0 !important; font-weight: 700 !important;
        border: none !important; border-bottom: none !important; box-shadow: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: block !important; height: 3px !important; background: #1A237E !important; }
    .stTabs [role="tablist"] { border-bottom: none !important; }
    .stTabs * { box-shadow: none !important; }
    .bmi-chart { background: white; padding: 1rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin: 1rem 0; }
    .info-box { background: #f8f9fa; border-left: 4px solid #1a237e; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
    .result-container {
        border-radius: 30px; padding: 40px 30px; min-height: 420px; height: 420px; max-height: 420px;
        border: 3px solid #333; text-align: center; display: flex; flex-direction: column;
        justify-content: center; animation: fadeIn 0.5s ease;
    }
    .result-container-high { border-color: #dc3545; background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%); }
    .result-container-low { border-color: #28a745; background: linear-gradient(135deg, #f0fff4 0%, #e8f5e9 100%); }
    .meme-container {
        border-radius: 30px; padding: 20px; min-height: 420px; height: 420px; max-height: 420px;
        border: 3px solid #333; text-align: center; animation: fadeIn 0.5s ease; background: white;
        display: flex; flex-direction: column; justify-content: space-between; align-items: center;
    }
    .meme-container img { max-height: 250px; object-fit: contain; width: 100%; border-radius: 15px; margin: 10px 0; flex-shrink: 0; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1a237e; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; }
    .validation-error-box { background: #fff0f0; border-left: 4px solid #f5576c; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .validation-error-box li { color: #dc3545; margin: 0.3rem 0; }
    .stFileUploader { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# ==================== Validation Functions ====================
def validate_pregnancies(value):
    if value < 0: 
        return False, "Pregnancies cannot be negative."
    elif value > 20: 
        return False, "Pregnancies should be 20 or less."
    return True, "Valid"

def validate_glucose(value):
    if value < 0: 
        return False, "Glucose level cannot be negative."
    elif value > 300: 
        return False, "Glucose level must be between 0 and 300 mg/dL."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

def validate_blood_pressure(value):
    if value < 0: 
        return False, "Blood pressure cannot be negative."
    elif value > 180: 
        return False, "Blood pressure must be between 0 and 180 mm Hg."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

def validate_skin_thickness(value):
    if value < 0: 
        return False, "Skin thickness cannot be negative."
    elif value > 99: 
        return False, "Skin thickness must be between 0 and 99 mm."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

def validate_insulin(value):
    if value == 0: 
        return True, "0 will be replaced with the dataset median."
    elif value < 15 or value > 900: 
        return False, "Insulin can be 0 or between 15 and 900."
    return True, "Valid"

def validate_bmi(value):
    if value < 0: 
        return False, "BMI cannot be negative."
    elif value > 70.0: 
        return False, "BMI must be between 0 and 70.0."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

def validate_diabetes_pedigree(value):
    if value < 0: 
        return False, "Diabetes pedigree cannot be negative."
    elif value > 2.5: 
        return False, "Diabetes pedigree must be between 0 and 2.5."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

def validate_age(value):
    if value < 0: 
        return False, "Age cannot be negative."
    elif value > 100: 
        return False, "Age must be between 0 and 100 years."
    elif value == 0:
        return True, "0 will be replaced with the dataset median."
    return True, "Valid"

# ==================== Load GIF Memes ====================
@st.cache_data
def load_memes():
    """Load memes from local folders"""
    high_risk_gifs = []
    low_risk_gifs = []
    
    # Check for organized folder structure (high_risk and low_risk subfolders)
    if os.path.exists("memes/high_risk"):
        high_risk_gifs = [os.path.join("memes/high_risk", f) 
                         for f in os.listdir("memes/high_risk") 
                         if any(f.lower().endswith(ext) for ext in ['.gif', '.jpg', '.jpeg', '.png', '.webp'])]
        # Sort to ensure consistent order
        high_risk_gifs.sort()
    
    if os.path.exists("memes/low_risk"):
        low_risk_gifs = [os.path.join("memes/low_risk", f) 
                        for f in os.listdir("memes/low_risk") 
                        if any(f.lower().endswith(ext) for ext in ['.gif', '.jpg', '.jpeg', '.png', '.webp'])]
        low_risk_gifs.sort()
    
    # If organized folders don't exist, try flat structure
    if not high_risk_gifs and not low_risk_gifs and os.path.exists("memes"):
        image_extensions = ['.gif', '.jpg', '.jpeg', '.png', '.webp']
        all_images = [os.path.join("memes", f) for f in os.listdir("memes")
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if all_images:
            high_keywords = ['high', 'bad', 'sad', 'negative', 'risk', 'danger', 'worried', 'scared']
            low_keywords = ['low', 'good', 'happy', 'positive', 'safe', 'healthy', 'great', 'celebration']
            
            high_risk_gifs = [img for img in all_images 
                            if any(k in os.path.basename(img).lower() for k in high_keywords)]
            low_risk_gifs = [img for img in all_images 
                           if any(k in os.path.basename(img).lower() for k in low_keywords)]
            
            if not high_risk_gifs and not low_risk_gifs and all_images:
                mid = len(all_images) // 2
                low_risk_gifs = all_images[:mid]
                high_risk_gifs = all_images[mid:]
            
            if not high_risk_gifs and all_images:
                high_risk_gifs = all_images[:len(all_images)//2]
                low_risk_gifs = all_images[len(all_images)//2:]
                if not high_risk_gifs:
                    high_risk_gifs = all_images[:1] if all_images else []
                    low_risk_gifs = all_images[1:] if len(all_images) > 1 else all_images
    
    return high_risk_gifs, low_risk_gifs

high_risk_gifs, low_risk_gifs = load_memes()

@st.cache_data(show_spinner=False)
def get_meme_img_html(selected_meme):
    """Load meme image with proper error handling"""
    if not selected_meme:
        return '<p style="color:#999;">No meme available</p>'
    
    try:
        if os.path.exists(selected_meme):
            ext = os.path.splitext(selected_meme)[1].lower().replace('.', '')
            mime_types = {
                'gif': 'image/gif',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'webp': 'image/webp',
                'svg': 'image/svg+xml'
            }
            mime = mime_types.get(ext, f'image/{ext}')
            
            with open(selected_meme, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            
            return f'<img src="data:{mime};base64,{b64}" alt="meme" style="max-width:100%; max-height:250px; object-fit:contain; border-radius:10px;">'
        else:
            return '<p style="color:#999;">Meme file not found</p>'
    except Exception as e:
        return f'<p style="color:#999;">Error loading meme</p>'

# ==================== Load Models ====================
@st.cache_resource
def load_models():
    missing = [f for f in ["diabetes_model.pkl", "scaler.pkl"] if not os.path.exists(f)]
    if missing:
        st.error(f"Missing required file(s): {', '.join(missing)}. "
                 "Run train_model.py and make sure these files are deployed alongside this app.")
        return None, None, None

    try:
        model = pickle.load(open("diabetes_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model/scaler: {str(e)}")
        return None, None, None

    # Try the real trained imputer first
    if os.path.exists("imputer.pkl"):
        try:
            imputer = pickle.load(open("imputer.pkl", "rb"))
            if not isinstance(imputer, dict):
                raise ValueError("imputer.pkl is an old sklearn-object format, not a median dict")
            st.success("Model, scaler, and imputer loaded successfully.")
            return model, scaler, imputer
        except Exception as e:
            st.warning(f"Could not load imputer.pkl ({str(e)}). Rebuilding from diabetes.csv...")

    # Fall back to rebuilding the imputer from the ORIGINAL dataset
    if os.path.exists("diabetes.csv"):
        st.info("Rebuilding imputer from diabetes.csv (imputer.pkl was missing or unusable)...")
        df = pd.read_csv("diabetes.csv")
        impute_cols_ = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]
        df[impute_cols_] = df[impute_cols_].replace(0, np.nan)
        imputer = df[impute_cols_].median().to_dict()
        pickle.dump(imputer, open("imputer.pkl", "wb"))
        st.success("imputer.pkl rebuilt from real training data.")
        return model, scaler, imputer

    st.error(
        "Cannot load or rebuild a real imputer: both imputer.pkl and diabetes.csv "
        "are missing from this deployment. Predictions are disabled until one of "
        "these is provided, so results are never based on fabricated data."
    )
    return None, None, None

model, scaler, imputer = load_models()
models_ready = all(x is not None for x in (model, scaler, imputer))

# IMPORTANT: impute_cols includes all columns that should be imputed
# Includes: Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction
# Excludes: Pregnancies (0 is valid), Age (0 should be caught by validation)
impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]

# ==================== Session State ====================
for key, default in [
    ('show_validation', False), ('show_result', False),
    ('prediction_result', None), ('selected_meme', None),
    ('input_mode', "Manual Input"), ('reset_counter', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def reset_app():
    st.session_state.show_validation = False
    st.session_state.show_result = False
    st.session_state.prediction_result = None
    st.session_state.selected_meme = None
    st.session_state.reset_counter += 1

def run_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    """Runs the ACTUAL trained pipeline: impute -> scale -> predict.
    Note: Pregnancies and Age are NOT imputed.
    Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction ARE imputed when 0."""
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    )
    # Only impute the columns in impute_cols (excludes Pregnancies and Age)
    input_data[impute_cols] = input_data[impute_cols].replace(0, float("nan"))
    input_data[impute_cols] = input_data[impute_cols].fillna(imputer)  # imputer is a {col: median} dict
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return result[0], probability[0]

def process_uploaded_file(uploaded_file):
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            try:
                import openpyxl
                engine = 'openpyxl' if file_extension == 'xlsx' else 'xlrd'
                try:
                    df = pd.read_excel(uploaded_file, engine=engine)
                except Exception as e:
                    df = pd.read_excel(uploaded_file)
            except ImportError:
                return None, "openpyxl is not installed. Please run: pip install openpyxl"
            except Exception as e:
                return None, f"Error reading Excel file: {str(e)}"
        else:
            return None, f"Unsupported file format: {file_extension}. Please upload CSV or Excel files."

        if df.empty:
            return None, "The uploaded file is empty."

        df.columns = df.columns.str.strip().str.replace(' ', '')
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing_cols = []
        for col in required_columns:
            if col not in df.columns:
                found = False
                for df_col in df.columns:
                    if df_col.lower() == col.lower():
                        df.rename(columns={df_col: col}, inplace=True)
                        found = True
                        break
                if not found:
                    missing_cols.append(col)
        if missing_cols:
            return None, f"Missing columns: {', '.join(missing_cols)}. Required columns: {', '.join(required_columns)}"

        first_row = df.iloc[0]
        data = {
            'Pregnancies': int(first_row['Pregnancies']) if pd.notna(first_row['Pregnancies']) else 0,
            'Glucose': int(first_row['Glucose']) if pd.notna(first_row['Glucose']) else 0,
            'BloodPressure': int(first_row['BloodPressure']) if pd.notna(first_row['BloodPressure']) else 0,
            'SkinThickness': int(first_row['SkinThickness']) if pd.notna(first_row['SkinThickness']) else 0,
            'Insulin': int(first_row['Insulin']) if pd.notna(first_row['Insulin']) else 0,
            'BMI': float(first_row['BMI']) if pd.notna(first_row['BMI']) else 0.0,
            'DiabetesPedigreeFunction': float(first_row['DiabetesPedigreeFunction']) if pd.notna(first_row['DiabetesPedigreeFunction']) else 0.0,
            'Age': int(first_row['Age']) if pd.notna(first_row['Age']) else 0,
        }
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# Create tabs
tab1, tab2 = st.tabs([" Diabetes Prediction", " BMI Calculator"])

# ==================== TAB 1: Diabetes Prediction ====================
with tab1:
    st.markdown('<div class="title-text"> Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Early detection can save lives. Enter patient details below for risk assessment.</div>', unsafe_allow_html=True)

    if not models_ready:
        st.warning("Prediction is disabled until the model files above are available.")

    st.markdown("### Select Input Method")
    col_mode1, col_mode2, col_mode3, col_mode4 = st.columns([1, 1, 1, 1])
    with col_mode1:
        if st.button(" Manual Input", use_container_width=True, key="btn_manual"):
            if st.session_state.input_mode != "Manual Input":
                st.session_state.input_mode = "Manual Input"
                reset_app()
                st.rerun()
    with col_mode2:
        if st.button(" Upload File", use_container_width=True, key="btn_upload"):
            if st.session_state.input_mode != "File Upload":
                st.session_state.input_mode = "File Upload"
                reset_app()
                st.rerun()

    if st.session_state.input_mode == "Manual Input":
        st.info(" Currently in Manual Input mode")
    else:
        st.info(" Currently in File Upload mode")

    st.markdown("---")

    # ==================== MANUAL INPUT MODE ====================
    if st.session_state.input_mode == "Manual Input":
        col1, col2 = st.columns([2, 1])

        with col1:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="section-header"> Personal Information</div>', unsafe_allow_html=True)

                Pregnancies = st.number_input(" Pregnancies", min_value=0, max_value=20, value=0, step=1,
                                               help="Number of times pregnant (0-20)", key=f"preg_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_pregnancies(Pregnancies)
                    st.markdown(f'<div class="validation-message {"validation-success" if v else "validation-error"}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                Glucose = st.number_input(" Glucose Level", min_value=0, max_value=300, value=0, step=1,
                                           help="Plasma glucose concentration (mg/dL) - Range: 0-300", key=f"gluc_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_glucose(Glucose)
                    cls = "validation-warning" if (v and Glucose == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                BloodPressure = st.number_input(" Blood Pressure", min_value=0, max_value=180, value=0, step=1,
                                                 help="Diastolic blood pressure (mm Hg) - Range: 0-180", key=f"bp_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_blood_pressure(BloodPressure)
                    cls = "validation-warning" if (v and BloodPressure == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                SkinThickness = st.number_input(" Skin Thickness", min_value=0, max_value=99, value=0, step=1,
                                                 help="Triceps skin fold thickness (mm) - Range: 0-99", key=f"skin_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_skin_thickness(SkinThickness)
                    cls = "validation-warning" if (v and SkinThickness == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="section-header"> Health Metrics</div>', unsafe_allow_html=True)

                Insulin = st.number_input(" Insulin", min_value=0, max_value=900, value=0, step=1,
                                           help="2-Hour serum insulin (mu U/ml) - 0 will be imputed with median", key=f"ins_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_insulin(Insulin)
                    cls = "validation-warning" if (v and Insulin == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                BMI = st.number_input(" BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1, format="%.1f",
                                       help="Body Mass Index - 0 will be imputed with median", key=f"bmi_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_bmi(BMI)
                    cls = "validation-warning" if (v and BMI == 0) else ("validation-success" if v else "validation-error")
                    msg = "0 will be imputed with median" if (v and BMI == 0) else m
                    st.markdown(f'<div class="validation-message {cls}">{msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                DiabetesPedigreeFunction = st.number_input(" Diabetes Pedigree", min_value=0.0, max_value=2.5, value=0.0,
                                                            step=0.001, format="%.3f", help="Diabetes pedigree function - Range: 0-2.5",
                                                            key=f"dpf_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_diabetes_pedigree(DiabetesPedigreeFunction)
                    cls = "validation-warning" if (v and DiabetesPedigreeFunction == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

                Age = st.number_input(" Age", min_value=0, max_value=100, value=0, step=1,
                                       help="Age in years - Range: 0-100", key=f"age_{st.session_state.reset_counter}")
                if st.session_state.show_validation:
                    v, m = validate_age(Age)
                    cls = "validation-warning" if (v and Age == 0) else ("validation-success" if v else "validation-error")
                    st.markdown(f'<div class="validation-message {cls}">{m}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-message-empty"></div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header"> Quick Stats</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p style="margin:0;"><strong>About Diabetes</strong></p>
                <p style="margin:0; font-size:0.9rem; color:#666;">
                    Diabetes is a chronic condition affecting how your body turns food into energy.
                    Early detection and proper management are crucial for preventing complications.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <p style="margin:0;"><strong>Risk Factors</strong></p>
                <ul style="font-size:0.9rem; color:#666; margin:0.5rem 0;">
                    <li>High glucose levels (> 140 mg/dL)</li>
                    <li>Elevated BMI (> 25)</li>
                    <li>Family history</li>
                    <li>Age over 45</li>
                    <li>High blood pressure</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(" Predict Diabetes Risk", use_container_width=True, key="predict", disabled=not models_ready)

        if predict_button:
            st.session_state.show_validation = True
            checks = [
                validate_pregnancies(Pregnancies), validate_glucose(Glucose),
                validate_blood_pressure(BloodPressure), validate_skin_thickness(SkinThickness),
                validate_insulin(Insulin), validate_bmi(BMI),
                validate_diabetes_pedigree(DiabetesPedigreeFunction), validate_age(Age),
            ]
            all_valid = all(v for v, _ in checks)

            if not all_valid:
                st.error(" Please fix the highlighted fields above before predicting.")
            elif not models_ready:
                st.error(" Model not loaded. Please check the model files.")
            else:
                try:
                    result, probability = run_prediction(
                        Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigreeFunction, Age
                    )
                    st.session_state.prediction_result = {"result": result, "probability": probability}
                    st.session_state.selected_meme = random.choice(high_risk_gifs if result == 1 else low_risk_gifs)
                    st.session_state.show_result = True
                    st.rerun()
                except Exception as e:
                    st.error(f" An error occurred during prediction: {str(e)}")

    # ==================== UPLOAD FILE MODE ====================
    else:
        st.markdown('<div class="section-header"> Upload Patient Data</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #666; margin-bottom: 1rem;">
            Upload a CSV or Excel file containing patient data. The first row will be used for prediction.
        </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing patient data. The first row will be used for prediction.",
            key=f"uploader_{st.session_state.reset_counter}", accept_multiple_files=False
        )

        if uploaded_file is not None:
            data, error = process_uploaded_file(uploaded_file)
            if error:
                st.error(f" {error}")
            elif data is not None:
                st.success(f" File '{uploaded_file.name}' processed successfully!")
                st.markdown("** Data Preview (First Row):**")
                st.dataframe(pd.DataFrame([data]), use_container_width=True)

                st.markdown("** Validation Results:**")
                validations = {
                    'Pregnancies': validate_pregnancies(data['Pregnancies']),
                    'Glucose': validate_glucose(data['Glucose']),
                    'BloodPressure': validate_blood_pressure(data['BloodPressure']),
                    'SkinThickness': validate_skin_thickness(data['SkinThickness']),
                    'Insulin': validate_insulin(data['Insulin']),
                    'BMI': validate_bmi(data['BMI']),
                    'DiabetesPedigreeFunction': validate_diabetes_pedigree(data['DiabetesPedigreeFunction']),
                    'Age': validate_age(data['Age']),
                }
                all_valid, has_errors, warning_fields = True, False, []
                for field, (is_valid, message) in validations.items():
                    if not is_valid:
                        st.markdown(f'<div class="validation-message validation-error"> {field}: {message}</div>', unsafe_allow_html=True)
                        has_errors, all_valid = True, False
                    elif "imputed" in message.lower() or "median" in message.lower():
                        st.markdown(f'<div class="validation-message validation-warning"> {field}: {message}</div>', unsafe_allow_html=True)
                        warning_fields.append(field)
                    else:
                        st.markdown(f'<div class="validation-message validation-success"> {field}: {message}</div>', unsafe_allow_html=True)

                if not has_errors:
                    if warning_fields:
                        st.info(f" Note: {', '.join(warning_fields)} will be imputed with median values.")
                    else:
                        st.markdown('<div class="validation-message validation-success"> All fields are valid!</div>', unsafe_allow_html=True)

                if has_errors:
                    st.warning(" Please fix the validation errors above. You can upload a corrected file or switch to manual input mode.")
                    st.markdown("---")
                    st.markdown("** Options to fix:**")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(" Upload New File", use_container_width=True, key="reset_upload"):
                            reset_app()
                            st.rerun()
                    with c2:
                        if st.button(" Switch to Manual Input", use_container_width=True, key="switch_manual"):
                            st.session_state.input_mode = "Manual Input"
                            reset_app()
                            st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    predict_upload_button = st.button(" Predict Risk from Uploaded Data", use_container_width=True,
                                                       key="predict_upload", disabled=not (all_valid and models_ready))

                if predict_upload_button and all_valid and models_ready:
                    try:
                        result, probability = run_prediction(
                            data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'],
                            data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], data['Age']
                        )
                        st.session_state.prediction_result = {"result": result, "probability": probability}
                        st.session_state.selected_meme = random.choice(high_risk_gifs if result == 1 else low_risk_gifs)
                        st.session_state.show_result = True
                        st.rerun()
                    except Exception as e:
                        st.error(f" An error occurred during prediction: {str(e)}")

    # ==================== DISPLAY RESULTS ====================
    if st.session_state.show_result and st.session_state.prediction_result is not None:
        result_data = st.session_state.prediction_result
        selected_meme = st.session_state.selected_meme

        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            if result_data["result"] == 1:
                st.markdown(f"""
                <div class="result-container result-container-high">
                    <h2 style="color: #dc3545;"> High Risk of Diabetes</h2>
                    <div style="font-size: 2.5rem; font-weight: 700; margin: 1rem 0; color: #dc3545;">
                        {result_data['probability'][1]:.1%}
                    </div>
                    <p style="font-size: 1.1rem; color: #666;">
                        Please consult a healthcare professional for proper diagnosis and management.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container result-container-low">
                    <h2 style="color: #28a745;"> Low Risk of Diabetes</h2>
                    <div style="font-size: 2.5rem; font-weight: 700; margin: 1rem 0; color: #28a745;">
                        {result_data['probability'][0]:.1%}
                    </div>
                    <p style="font-size: 1.1rem; color: #666;">
                        Continue maintaining a healthy lifestyle!
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            img_html = get_meme_img_html(selected_meme)
            caption = "Don't worry! Early detection is important." if result_data["result"] == 1 else "Keep up the healthy lifestyle!"
            st.markdown(f"""
            <div class="meme-container">
                <h4 style="margin-bottom: 15px; color: #333;"> Today's Meme</h4>
                {img_html}
                <hr style="width:100%; border: none; border-top: 1px solid #eee; margin: 15px 0;">
                <p style="color:#666; font-size:0.9rem; margin:0;">{caption}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" Reset Prediction", use_container_width=True, key="reset_prediction"):
                reset_app()
                st.rerun()

# ==================== TAB 2: BMI Calculator ====================
with tab2:
    st.markdown('<div class="title-text"> BMI Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Calculate your Body Mass Index and understand your health status</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"> Enter Your Measurements</div>', unsafe_allow_html=True)
        unit_system = st.radio("Select Unit System", ["Metric (kg/cm)", "Imperial (lbs/in)"], horizontal=True, key="unit_system")
        st.markdown("---")

        # Initialize variables
        height_cm = 0.0
        height_in = 0.0
        bmi = 0.0

        if unit_system == "Metric (kg/cm)":
            weight = st.number_input(" Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5, format="%.1f", key="weight_metric")
            height_cm = st.number_input(" Height (cm)", min_value=50.0, max_value=300.0, value=170.0, step=0.5, format="%.1f", key="height_metric")
            if height_cm > 0:
                bmi = weight / ((height_cm / 100) ** 2)
        else:
            weight = st.number_input(" Weight (lbs)", min_value=1.0, max_value=660.0, value=154.0, step=0.5, format="%.1f", key="weight_imperial")
            height_in = st.number_input(" Height (inches)", min_value=20.0, max_value=120.0, value=67.0, step=0.5, format="%.1f", key="height_imperial")
            if height_in > 0:
                bmi = (weight / (height_in ** 2)) * 703

        st.markdown("<br>", unsafe_allow_html=True)
        calculate_bmi = st.button(" Calculate BMI", use_container_width=True, key="calc_bmi")

    with col2:
        st.markdown('<div class="section-header"> BMI Result</div>', unsafe_allow_html=True)

        if calculate_bmi and weight > 0 and (height_cm > 0 or height_in > 0):
            if bmi < 18.5:
                category, color_class = "Underweight", "bmi-underweight"
                description = "You may need to gain weight. Consult a healthcare professional for guidance."
            elif 18.5 <= bmi < 25:
                category, color_class = "Normal Weight", "bmi-normal"
                description = "Great job! Maintain your healthy lifestyle."
            elif 25 <= bmi < 30:
                category, color_class = "Overweight", "bmi-overweight"
                description = "Consider adopting a healthier diet and increasing physical activity."
            else:
                category, color_class = "Obese", "bmi-obese"
                description = "Please consult a healthcare professional for a comprehensive health plan."

            st.markdown(f"""
            <div class="{color_class}">
                <div class="bmi-category">{category}</div>
                <div class="bmi-value">{bmi:.1f}</div>
                <div class="bmi-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1rem;"> BMI Scale</div>', unsafe_allow_html=True)
            bmi_display = min(bmi, 40)
            bmi_percentage = (bmi_display / 40) * 100
            st.markdown(f"""
            <div class="bmi-chart">
                <div style="position: relative; height: 30px; background: #f0f0f0; border-radius: 15px; overflow: hidden;">
                    <div style="position: absolute; left: 0; top: 0; height: 100%; width: {min(bmi_percentage, 100)}%;
                         background: linear-gradient(90deg, #1a237e, #0d47a1, #1565c0, #1a237e);
                         border-radius: 15px; transition: width 0.8s ease;">
                    </div>
                    <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
                         font-weight: 600; font-size: 0.8rem; color: white; text-shadow: 0 0 10px rgba(0,0,0,0.5);">
                        {bmi:.1f}
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.7rem; color: #1a237e;">
                    <span>Underweight</span><span>Normal</span><span>Overweight</span><span>Obese</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(" Detailed BMI Information", expanded=True):
                st.markdown(f"""
                **Your BMI:** {bmi:.1f}

                **Category:** {category}

                **Health Implications:**
                - Underweight (< 18.5): May indicate malnutrition, eating disorders, or other health issues
                - Normal (18.5 - 24.9): Healthy weight range for most adults
                - Overweight (25 - 29.9): Increased risk of health problems
                - Obese (>= 30): High risk of health problems including diabetes, heart disease, and more

                **Note:** BMI is a screening tool and doesn't account for muscle mass, bone density, or overall body composition.
                """)
                if bmi < 18.5:
                    st.info("""
                    **Recommendations for Underweight:**
                    - Eat nutrient-rich foods more frequently
                    - Strength training to build muscle mass
                    - Consult a healthcare professional
                    - Include healthy fats in your diet
                    - Increase calorie intake with healthy options
                    """)
                elif 18.5 <= bmi < 25:
                    st.success("""
                    **Maintain Your Healthy Weight:**
                    - Continue balanced nutrition
                    - Regular physical activity (150 min/week)
                    - Adequate sleep (7-9 hours)
                    - Stress management
                    - Stay hydrated
                    """)
                elif 25 <= bmi < 30:
                    st.warning("""
                    **Tips for Weight Management:**
                    - Reduce calorie intake gradually
                    - Increase physical activity
                    - Choose whole foods over processed
                    - Stay hydrated
                    - Monitor portion sizes
                    - Regular health checkups
                    """)
                else:
                    st.error("""
                    **Important Actions to Take:**
                    - Schedule a comprehensive health checkup
                    - Consult a nutritionist for a personalized diet plan
                    - Start with moderate exercise (consult your doctor first)
                    - Monitor blood pressure and blood sugar levels
                    - Follow medical advice and prescribed treatments
                    - Consider stress management techniques
                    """)
        else:
            st.info(" Enter your weight and height, then click 'Calculate BMI' to see your results.")
            st.markdown('<div class="section-header" style="margin-top:1rem;"> BMI Reference Chart</div>', unsafe_allow_html=True)
            bmi_data = {
                "Category": ["Underweight", "Normal", "Overweight", "Obese"],
                "BMI Range": ["< 18.5", "18.5 - 24.9", "25 - 29.9", ">= 30"],
                "Status": [" Needs Attention", " Healthy", " Caution", " High Risk"],
            }
            st.table(pd.DataFrame(bmi_data).style.hide(axis='index'))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #1a237e; font-size: 0.8rem; padding: 1rem;">
    This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.<br>
    Memes are for entertainment purposes and should not affect medical decisions.
</div>
""", unsafe_allow_html=True)
