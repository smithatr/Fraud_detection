import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import hashlib
import os
import base64
from PIL import Image
from io import BytesIO
import importlib
import time
import threading
import random
import queue
import requests
from datetime import datetime
import numpy as np

# Create a queue for simulated streaming data
if 'data_queue' not in st.session_state:
    st.session_state.data_queue = queue.Queue()

# Background image paths (update these paths to your actual image locations)
BACKGROUND_IMAGES = {
    "Login": "images/login_bg.jpg",
    "Signup": "images/signup_bg.jpg",
    "Home": "images/home_bg.jpg",
    "Visualizations": "images/viz_bg.jpg",
    "History": "images/history_bg.jpg",
    "Manual Input": "images/manual_bg.jpg",
    "Feature Selection": "images/feature_bg.jpg",
    "About": "images/about_bg.jpg",
    "Streaming": "images/streaming_bg.jpg"
}

# Ensure the images directory exists
os.makedirs("images", exist_ok=True)

# Function to download placeholder images if they don't exist
def download_placeholder_images():
    image_urls = {
        "login_bg.jpg": "https://images.unsplash.com/photo-1579621970588-a35d0e7ab9b6",
        "signup_bg.jpg": "https://images.unsplash.com/photo-1553877522-43269d4ea984",
        "home_bg.jpg": "https://images.unsplash.com/photo-1589758438368-0ad531db3366",
        "viz_bg.jpg": "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        "history_bg.jpg": "https://images.unsplash.com/photo-1460925895917-afdab827c52f",
        "manual_bg.jpg": "https://images.unsplash.com/photo-1579621970795-87facc2f976d",
        "feature_bg.jpg": "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40",
        "about_bg.jpg": "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d",
        "streaming_bg.jpg": "https://images.unsplash.com/photo-1519389950473-47ba0277781c"
    }
    
    for img_name, url in image_urls.items():
        file_path = os.path.join("images", img_name)
        if not os.path.exists(file_path):
            try:
                response = requests.get(f"{url}?q=85&w=1920&auto=format", stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    st.success(f"Downloaded {img_name}")
                else:
                    st.warning(f"Failed to download {img_name}")
            except Exception as e:
                st.error(f"Error downloading {img_name}: {str(e)}")

# Load trained model and scaler
try:
    model = joblib.load("fraud_detection_model222.pkl")
    scaler = joblib.load("scaler222.pkl")
except:
    st.error("Model files not found. Using placeholder functionality.")
    model = None
    scaler = None

# Define required feature columns
required_features = ['Transaction_Details', 'Cardholder_Information', 'Device_and_Network_Information',
                     'Historical_Data', 'Behavioral_Data', 'Security_Features', 'External_Data']

# Function to get local background image
def get_base64_of_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

def add_bg_from_local(image_path):
    try:
        bin_str = get_base64_of_image(image_path)
        if bin_str:
            return f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """
        else:
            return """
            <style>
            .stApp {
                background-color: #f0f2f6;
            }
            </style>
            """
    except:
        return """
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        </style>
        """

# Custom CSS for styling
def apply_custom_styling():
    st.markdown("""
        <style>
        .main-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .stFileUploader>div>div>div>button {
            background-color: #008CBA;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        }
        .stMarkdown h1 {
            color: #2E4057;
            font-size: 36px;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .stMarkdown h2 {
            color: #1A535C;
            font-size: 28px;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .stMarkdown h3 {
            color: #4ECDC4;
            font-size: 22px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #2E4057;
            color: white;
            border-right: 1px solid #ddd;
        }
        .user-info {
            padding: 15px;
            background-color: rgba(46, 64, 87, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
        }
        .stSelectbox {
            border-radius: 10px;
        }
        .stNumberInput {
            border-radius: 10px;
        }
        .stSlider {
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .real-time-container {
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .real-time-header {
            color: #2E4057;
            font-size: 24px;
            margin-bottom: 15px;
        }
        .transaction-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s;
            border-left: 5px solid #4CAF50;
        }
        .transaction-card.fraud {
            border-left: 5px solid #FF6B6B;
        }
        .transaction-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-indicator.active {
            background-color: #4CAF50;
            animation: pulse 1.5s infinite;
        }
        .status-indicator.inactive {
            background-color: #FF6B6B;
        }
        @keyframes pulse {
            0% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
            }
        }
        </style>
        """, unsafe_allow_html=True)

# Function to generate a simulated transaction
def generate_transaction():
    transaction = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Transaction_ID': f"T{random.randint(100000, 999999)}",
        'Transaction_Details': random.uniform(100, 10000),
        'Cardholder_Information': random.choice(["Verified", "Unverified"]),
        'Device_and_Network_Information': random.choice(["Secure", "Unknown", "Compromised"]),
        'Historical_Data': random.uniform(500, 2000),
        'Behavioral_Data': random.uniform(0, 1),
        'Security_Features': random.choice(["Enabled", "Partial", "Disabled"]),
        'External_Data': random.choice(["Low Risk", "Moderate Risk", "High Risk"])
    }
    return transaction

# Function to simulate fraud probability based on transaction features
def simulate_fraud_probability(transaction):
    # Base probability
    base_prob = 0.05
    
    # Increase probability based on risk factors
    if transaction['Cardholder_Information'] == "Unverified":
        base_prob += 0.2
    
    if transaction['Device_and_Network_Information'] == "Compromised":
        base_prob += 0.3
    elif transaction['Device_and_Network_Information'] == "Unknown":
        base_prob += 0.1
    
    if transaction['Security_Features'] == "Disabled":
        base_prob += 0.25
    elif transaction['Security_Features'] == "Partial":
        base_prob += 0.1
    
    if transaction['External_Data'] == "High Risk":
        base_prob += 0.3
    elif transaction['External_Data'] == "Moderate Risk":
        base_prob += 0.15
    
    # Add some randomness
    base_prob += random.uniform(-0.05, 0.05)
    
    # Cap the probability between 0 and 1
    return max(0, min(1, base_prob))

# Function to start the data streaming thread
def start_data_stream():
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False
    
    if not st.session_state.stream_active:
        st.session_state.stream_active = True
        threading.Thread(target=generate_stream_data, daemon=True).start()

# Function to stop the data streaming
def stop_data_stream():
    st.session_state.stream_active = False

# Function to generate streaming data in a separate thread
def generate_stream_data():
    while st.session_state.stream_active:
        # Generate a new transaction
        transaction = generate_transaction()
        
        # Calculate fraud probability
        fraud_prob = simulate_fraud_probability(transaction)
        fraud_pred = 1 if fraud_prob > 0.5 else 0
        
        # Add predictions to the transaction
        transaction['Fraud_Probability'] = fraud_prob
        transaction['Predicted_Fraud'] = fraud_pred
        transaction['Alerts_and_Notifications'] = "🔴 High Risk Alert" if fraud_prob > 0.8 else "🟡 Medium Risk" if fraud_prob > 0.5 else "🟢 Low Risk"
        transaction['Historical_Insights'] = "📊 Previous fraud cases analyzed"
        transaction['User_Actions'] = "🛑 Review Required" if fraud_pred == 1 else "✅ Approved"
        
        # Add to the queue
        st.session_state.data_queue.put(transaction)
        
        # Control the generation rate (transactions per second)
        time.sleep(random.uniform(1, 3))

# Authentication functions
def generate_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(username, password, password_hashes):
    if username in password_hashes:
        return password_hashes[username] == generate_hash(password)
    return False

def save_user(username, password):
    if 'password_hashes' not in st.session_state:
        st.session_state.password_hashes = {}
    st.session_state.password_hashes[username] = generate_hash(password)
    return True

# Initialize session state variables
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"
if 'password_hashes' not in st.session_state:
    st.session_state.password_hashes = {"admin": generate_hash("admin123")}  # Default user
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []
if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False

# Preprocessing function
def preprocess_data(df):
    """Preprocess uploaded data for model prediction."""
    try:
        df = df[required_features]
        for col in required_features:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        if scaler:
            df = scaler.transform(df)
        return df
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        # Return dummy data for demonstration
        import numpy as np
        return np.random.rand(len(df), len(required_features))
# Function to handle login
def login_page():
    # Apply background for login page
    if BACKGROUND_IMAGES.get("Login") and os.path.exists(BACKGROUND_IMAGES["Login"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Login"]), unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-container">
            <h1 style="text-align: center;">🔐 Fraud Detection System Login</h1>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            username = st.text_input("👤 Username")
            password = st.text_input("🔑 Password", type="password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                login_button = st.button("Login")
            with col2:
                signup_redirect = st.button("Create Account")
            st.markdown('</div>', unsafe_allow_html=True)
                
            if login_button:
                if check_password(username, password, st.session_state.password_hashes):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.current_page = "Home"
                    st.rerun()
                else:
                    st.error("⚠️ Invalid username or password")
                    
            if signup_redirect:
                st.session_state.current_page = "Signup"
                st.rerun()

# Function to handle signup
def signup_page():
    # Apply background for signup page
    if BACKGROUND_IMAGES.get("Signup") and os.path.exists(BACKGROUND_IMAGES["Signup"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Signup"]), unsafe_allow_html=True)
    
    # Center the signup form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-container">
            <h1 style="text-align: center;">📝 Create Your Account</h1>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            new_username = st.text_input("👤 Choose Username")
            new_password = st.text_input("🔑 Choose Password", type="password")
            confirm_password = st.text_input("🔁 Confirm Password", type="password")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                signup_button = st.button("Create Account")
            with col2:
                login_redirect = st.button("Back to Login")
            st.markdown('</div>', unsafe_allow_html=True)
                
            if signup_button:
                if new_username in st.session_state.password_hashes:
                    st.error("⚠️ Username already exists!")
                elif new_password != confirm_password:
                    st.error("⚠️ Passwords do not match!")
                else:
                    if save_user(new_username, new_password):
                        st.success("✅ Account created successfully! Please login.")
                        st.session_state.current_page = "Login"
                        st.rerun()
                        
            if login_redirect:
                st.session_state.current_page = "Login"
                st.rerun()

# Modified Navigation
def show_navigation():
    with st.sidebar:
        st.title("🚨 Fraud Detection")
        
        # Show user info if logged in
        if st.session_state.logged_in:
            st.markdown(f"""
            <div class="user-info">
                <h3>👤 Welcome, {st.session_state.username}!</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Navigation options
        pages = {
            "🏠 Dashboard": "Home",
            "📊 Data Visualization": "Visualizations",
            "📡 Real-time Monitoring": "Streaming",
            "📜 Transaction History": "History",
            "✍️ Manual Analysis": "Manual Input",
           # "⚙️ Configuration": "Feature Selection",
            "ℹ️ About System": "About"
        }
        
        for label, page_name in pages.items():
            if st.sidebar.button(label, key=f"nav_{page_name}"):
                st.session_state.current_page = page_name
                st.rerun()
        
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.current_page = "Login"
            # Make sure to stop the stream when logging out
            if st.session_state.stream_active:
                stop_data_stream()
            st.rerun()

# Home Page
def home_page():
    # Apply background
    if BACKGROUND_IMAGES.get("Home") and os.path.exists(BACKGROUND_IMAGES["Home"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Home"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>🚨 Fraud Detection Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Active Monitoring", "Enabled" if st.session_state.stream_active else "Disabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        total_transactions = len(st.session_state.predictions) + len(st.session_state.real_time_data)
        st.metric("Total Transactions", total_transactions)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if hasattr(st.session_state, 'real_time_data') and st.session_state.real_time_data:
            fraud_count = sum(1 for t in st.session_state.real_time_data if t.get('Predicted_Fraud') == 1)
            fraud_rate = (fraud_count / len(st.session_state.real_time_data)) * 100 if st.session_state.real_time_data else 0
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        else:
            st.metric("Fraud Rate", "0.00%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h2>Welcome to the Fraud Detection System!</h2>
        <p>Upload your transaction data in CSV format, or enable real-time monitoring to detect fraud in streaming transactions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Transaction Data")
        uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📡 Real-time Monitoring")
        
        if st.session_state.stream_active:
            if st.button("🛑 Stop Monitoring"):
                stop_data_stream()
                st.success("Real-time monitoring stopped!")
                st.rerun()
            st.markdown("<p><span class='status-indicator active'></span> Actively monitoring transactions</p>", unsafe_allow_html=True)
        else:
            if st.button("▶️ Start Monitoring"):
                start_data_stream()
                st.success("Real-time monitoring started!")
                st.rerun()
            st.markdown("<p><span class='status-indicator inactive'></span> Monitoring inactive</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if set(required_features).issubset(df.columns) or model is None:
            st.success("✅ Data Loaded Successfully!")
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            try:
                # Preprocess data and make predictions
                X = preprocess_data(df)
                
                if model:
                    fraud_prob = model.predict_proba(X)[:, 1]
                    fraud_pred = model.predict(X)
                else:
                    # Demo mode if model not available
                    import numpy as np
                    fraud_prob = np.random.rand(len(df))
                    fraud_pred = (fraud_prob > 0.5).astype(int)
                
                # Add predictions to the dataframe
                df['Fraud_Probability'] = fraud_prob
                df['Predicted_Fraud'] = fraud_pred
                df['Alerts_and_Notifications'] = df['Fraud_Probability'].apply(
                    lambda x: "🔴 High Risk Alert" if x > 0.8 else "🟡 Medium Risk" if x > 0.5 else "🟢 Low Risk")
                df['Historical_Insights'] = "📊 Previous fraud cases analyzed"
                df['User_Actions'] = df['Predicted_Fraud'].apply(
                    lambda x: "🛑 Review Required" if x == 1 else "✅ Approved")
                df['Performance_Metrics'] = "📈 Model Accuracy: 95% (Example)"
                
                # Store predictions in session state
                st.session_state.predictions = df
                
                # Display results
                st.markdown("### 📊 Prediction Results:")
                st.dataframe(df[['Fraud_Probability', 'Predicted_Fraud', 'Alerts_and_Notifications', 
                                'Historical_Insights', 'User_Actions', 'Performance_Metrics']].head(20))
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Fraud Detected", df['Predicted_Fraud'].sum())
                with col3:
                    st.metric("Fraud Rate", f"{(df['Predicted_Fraud'].mean() * 100):.2f}%")
                
                # Download option
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions", csv_output, "fraud_predictions.csv", "text/csv")
                
            except Exception as e:
                st.error(f"❌ An error occurred during processing: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Uploaded file does not contain the required columns. Please ensure the file includes the following columns:")
            st.write(required_features)
import time
import random
from datetime import datetime

def streaming_page():
    # Apply background
    if BACKGROUND_IMAGES.get("Streaming") and os.path.exists(BACKGROUND_IMAGES["Streaming"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Streaming"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>📡 Real-time Transaction Monitoring</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Control panel in a card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.session_state.stream_active:
            if st.button("🛑 Stop Data Stream"):
                st.session_state.stream_active = False
                st.success("Data stream stopped!")
                st.rerun()
            status_text = "<span class='status-indicator active'></span> Stream Active"
        else:
            if st.button("▶️ Start Data Stream"):
                st.session_state.stream_active = True
                st.success("Data stream started!")
                st.rerun()
            status_text = "<span class='status-indicator inactive'></span> Stream Inactive"
    
    with col2:
        alert_threshold = st.slider("🚨 Alert Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown(f"<p>{status_text}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create columns for real-time transactions and alerts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="real-time-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="real-time-header">📊 Live Transaction Feed</h2>', unsafe_allow_html=True)
        
        # Create a placeholder for the transaction feed
        transaction_feed = st.empty()
        
        # Create a placeholder for the chart
        chart_placeholder = st.empty()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="real-time-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="real-time-header">🚨 Fraud Alerts</h2>', unsafe_allow_html=True)
        
        # Create a placeholder for the alerts
        alerts_placeholder = st.empty()
        
        # Show statistics
        st.markdown("### 📈 Real-time Statistics")
        
        if 'real_time_data' in st.session_state and st.session_state.real_time_data:
            total_transactions = len(st.session_state.real_time_data)
            fraud_count = sum(1 for t in st.session_state.real_time_data if t.get('Predicted_Fraud') == 1)
            fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transactions", total_transactions)
            with col2:
                st.metric("Fraud Detected", fraud_count)
            
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        else:
            st.metric("Total Transactions", 0)
            st.metric("Fraud Detected", 0)
            st.metric("Fraud Rate", "0.00%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate random data every 10 seconds if stream is active
    if st.session_state.stream_active:
        if 'real_time_data' not in st.session_state:
            st.session_state.real_time_data = []
        
        # Generate a new transaction every 10 seconds
        if 'last_generated' not in st.session_state or (time.time() - st.session_state.last_generated) >= 5:
            transaction = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Transaction_ID': f"T{random.randint(100000, 999999)}",
                'Transaction_Details': random.uniform(100, 10000),
                'Cardholder_Information': random.choice(["Verified", "Unverified"]),
                'Device_and_Network_Information': random.choice(["Secure", "Unknown", "Compromised"]),
                'Historical_Data': random.uniform(500, 2000),
                'Behavioral_Data': random.uniform(0, 1),
                'Security_Features': random.choice(["Enabled", "Partial", "Disabled"]),
                'External_Data': random.choice(["Low Risk", "Moderate Risk", "High Risk"])
            }
            
            # Simulate fraud probability
            fraud_prob = simulate_fraud_probability(transaction)
            fraud_pred = 1 if fraud_prob > 0.5 else 0
            
            # Add predictions to the transaction
            transaction['Fraud_Probability'] = fraud_prob
            transaction['Predicted_Fraud'] = fraud_pred
            transaction['Alerts_and_Notifications'] = "🔴 High Risk Alert" if fraud_prob > 0.8 else "🟡 Medium Risk" if fraud_prob > 0.5 else "🟢 Low Risk"
            transaction['User_Actions'] = "🛑 Review Required" if fraud_pred == 1 else "✅ Approved"
            
            # Add to real-time data
            st.session_state.real_time_data.append(transaction)
            st.session_state.last_generated = time.time()
        
        # Display the latest transactions
        if st.session_state.real_time_data:
            latest_transactions = st.session_state.real_time_data[-10:]  # Show last 10 transactions
            transaction_feed.markdown("".join([
                f"""
                <div class="transaction-card {'fraud' if t['Predicted_Fraud'] == 1 else ''}">
                    <p><strong>🕒 {t['timestamp']}</strong></p>
                    <p>Transaction ID: {t['Transaction_ID']}</p>
                    <p>Amount: ${t['Transaction_Details']:.2f}</p>
                    <p>Risk Level: {t['Alerts_and_Notifications']}</p>
                    <p>Action: {t['User_Actions']}</p>
                </div>
                """ for t in latest_transactions
            ]), unsafe_allow_html=True)
        
        # Update the chart
        if st.session_state.real_time_data:
            df = pd.DataFrame(st.session_state.real_time_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Resample only numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            resampled_df = df[numeric_columns].resample('10S').mean().reset_index()
            
            fig = px.line(resampled_df, x='timestamp', y='Fraud_Probability', 
                          title="Fraud Probability Over Time", labels={'Fraud_Probability': 'Fraud Probability'})
            fig.update_layout(xaxis_title="Time", yaxis_title="Fraud Probability", template="plotly_white")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Update the alerts
        if st.session_state.real_time_data:
            high_risk_alerts = [t for t in st.session_state.real_time_data if t['Fraud_Probability'] > alert_threshold]
            if high_risk_alerts:
                alerts_placeholder.markdown("".join([
                    f"""
                    <div class="transaction-card fraud">
                        <p><strong>🚨 High Risk Alert</strong></p>
                        <p>Transaction ID: {alert['Transaction_ID']}</p>
                        <p>Amount: ${alert['Transaction_Details']:.2f}</p>
                        <p>Fraud Probability: {alert['Fraud_Probability']:.2%}</p>
                        <p>Action: {alert['User_Actions']}</p>
                    </div>
                    """ for alert in high_risk_alerts[-5:]  # Show last 5 high-risk alerts
                ]), unsafe_allow_html=True)
            else:
                alerts_placeholder.markdown("<p>No high-risk alerts detected.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p>Stream is inactive. Start the stream to monitor transactions.</p>", unsafe_allow_html=True)
def visualizations_page():
    
    # Apply background
    if BACKGROUND_IMAGES.get("Visualizations") and os.path.exists(BACKGROUND_IMAGES["Visualizations"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Visualizations"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>📊 Data Visualizations</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.real_time_data and st.session_state.predictions.empty:
        st.warning("No data available for visualization. Please upload data or enable real-time monitoring.")
        return
    
    # Combine real-time data and predictions
    if st.session_state.real_time_data:
        df = pd.DataFrame(st.session_state.real_time_data)
    else:
        df = st.session_state.predictions
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Fraud Probability Distribution")
    fig = px.histogram(df, x='Fraud_Probability', nbins=20, title="Fraud Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Fraud by Category")
    category_col = st.selectbox("Select a category to analyze", ['Cardholder_Information', 'Device_and_Network_Information', 'Security_Features', 'External_Data'])
    fraud_by_category = df.groupby(category_col)['Predicted_Fraud'].mean().reset_index()
    fig = px.bar(fraud_by_category, x=category_col, y='Predicted_Fraud', title=f"Fraud Rate by {category_col}")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    #st.markdown("### 📉 Fraud Over Time")
    #if 'timestamp' in df.columns:
        #df['timestamp'] = pd.to_datetime(df['timestamp'])
        #df = df.set_index('timestamp')
        #resampled_df = df.resample('1T').mean().reset_index()  # Resample by minute
        #fig = px.line(resampled_df, x='timestamp', y='Fraud_Probability', title="Fraud Probability Over Time")
        #st.plotly_chart(fig, use_container_width=True)
   # else:
        #st.warning("Timestamp data not available for time-based visualization.")
    #st.markdown('</div>', unsafe_allow_html=True)


def history_page():
    # Apply background
    if BACKGROUND_IMAGES.get("History") and os.path.exists(BACKGROUND_IMAGES["History"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["History"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>📜 Transaction History</h1>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.predictions.empty and not st.session_state.real_time_data:
        st.warning("No historical data available. Please upload data or enable real-time monitoring.")
        return
    
    # Combine real-time data and predictions
    if st.session_state.real_time_data:
        df = pd.DataFrame(st.session_state.real_time_data)
    else:
        df = st.session_state.predictions
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Transaction Data")
    st.dataframe(df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "transaction_history.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


def manual_input_page():
    # Apply background
    if BACKGROUND_IMAGES.get("Manual Input") and os.path.exists(BACKGROUND_IMAGES["Manual Input"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["Manual Input"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>✍️ Manual Transaction Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Transaction Details")
    
    with st.form("manual_input_form"):
        transaction_details = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        cardholder_info = st.selectbox("Cardholder Information", ["Verified", "Unverified"])
        device_info = st.selectbox("Device and Network Information", ["Secure", "Unknown", "Compromised"])
        historical_data = st.number_input("Historical Data", min_value=0.0, value=500.0)
        behavioral_data = st.number_input("Behavioral Data", min_value=0.0, max_value=1.0, value=0.5)
        security_features = st.selectbox("Security Features", ["Enabled", "Partial", "Disabled"])
        external_data = st.selectbox("External Data", ["Low Risk", "Moderate Risk", "High Risk"])
        
        submitted = st.form_submit_button("Analyze Transaction")
    
    if submitted:
        transaction = {
            'Transaction_Details': transaction_details,
            'Cardholder_Information': cardholder_info,
            'Device_and_Network_Information': device_info,
            'Historical_Data': historical_data,
            'Behavioral_Data': behavioral_data,
            'Security_Features': security_features,
            'External_Data': external_data
        }
        
        # Preprocess and predict
        df = pd.DataFrame([transaction])
        X = preprocess_data(df)
        
        if model:
            fraud_prob = model.predict_proba(X)[:, 1][0]
            fraud_pred = model.predict(X)[0]
        else:
            fraud_prob = random.uniform(0, 1)
            fraud_pred = 1 if fraud_prob > 0.5 else 0
        
        st.markdown("### 📊 Analysis Results")
        st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        st.metric("Predicted Fraud", "Yes" if fraud_pred == 1 else "No")
        
        if fraud_pred == 1:
            st.error("🚨 High Risk: This transaction is likely fraudulent.")
        else:
            st.success("✅ Low Risk: This transaction appears legitimate.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def about_page():
    # Apply background
    if BACKGROUND_IMAGES.get("About") and os.path.exists(BACKGROUND_IMAGES["About"]):
        st.markdown(add_bg_from_local(BACKGROUND_IMAGES["About"]), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-container">
        <h1>ℹ️ About the Fraud Detection System</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 Overview
    This system is designed to detect fraudulent transactions in real-time using machine learning models. 
    It provides a user-friendly interface for monitoring, analyzing, and visualizing transaction data.
    
    ### 🔧 Features
    - Real-time transaction monitoring
    - Manual transaction analysis
    - Interactive visualizations
    - Historical data review
    - Fraud probability predictions
    
    ### 🛠️ Technologies Used
    - **Streamlit**: For the web interface
    - **Scikit-learn**: For machine learning models
    - **Plotly**: For interactive visualizations
    - **Pandas**: For data manipulation
    
    ### 📧 Contact
    For support or inquiries, please contact [support@frauddetection.com](mailto:support@frauddetection.com).
    """)
    st.markdown('</div>', unsafe_allow_html=True)
def main():
    # Apply custom styling
    apply_custom_styling()
    
    # Download placeholder images if they don't exist
    download_placeholder_images()
    
    # Show navigation sidebar
    show_navigation()
    
    # Page routing
    if st.session_state.current_page == "Login":
        login_page()
    elif st.session_state.current_page == "Signup":
        signup_page()
    elif st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Visualizations":
        visualizations_page()
    elif st.session_state.current_page == "Streaming":
        streaming_page()
    elif st.session_state.current_page == "History":
        history_page()
    elif st.session_state.current_page == "Manual Input":
        manual_input_page()
    elif st.session_state.current_page == "About":
        about_page()

# Run the app
if __name__ == "__main__":
    main()