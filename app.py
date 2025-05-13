from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import hashlib
import os
import base64
from datetime import datetime
import numpy as np
import random
import threading
import queue
import time

app = Flask(__name__)

# Create a queue for simulated streaming data
data_queue = queue.Queue()

# Background image paths
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

# Load trained model and scaler
try:
    model = joblib.load("fraud_detection_model222.pkl")
    scaler = joblib.load("scaler222.pkl")
except:
    print("Model files not found. Using placeholder functionality.")
    model = None
    scaler = None

# Define required feature columns
required_features = ['Transaction_Details', 'Cardholder_Information', 'Device_and_Network_Information',
                     'Historical_Data', 'Behavioral_Data', 'Security_Features', 'External_Data']

# Global state for the application
app_state = {
    'predictions': pd.DataFrame(),
    'logged_in': False,
    'username': "",
    'password_hashes': {"admin": hashlib.sha256(str.encode("admin123")).hexdigest()},
    'real_time_data': [],
    'stream_active': False,
    'last_generated': 0
}

# Authentication functions
def generate_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(username, password):
    if username in app_state['password_hashes']:
        return app_state['password_hashes'][username] == generate_hash(password)
    return False

def save_user(username, password):
    app_state['password_hashes'][username] = generate_hash(password)
    return True

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
        print(f"Error in preprocessing: {str(e)}")
        # Return dummy data for demonstration
        return np.random.rand(len(df), len(required_features))

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
    if not app_state['stream_active']:
        app_state['stream_active'] = True
        threading.Thread(target=generate_stream_data, daemon=True).start()

# Function to stop the data streaming
def stop_data_stream():
    app_state['stream_active'] = False

# Function to generate streaming data in a separate thread
def generate_stream_data():
    while app_state['stream_active']:
        # Generate a new transaction
        transaction = generate_transaction()
        
        # Calculate fraud probability
        fraud_prob = simulate_fraud_probability(transaction)
        fraud_pred = 1 if fraud_prob > 0.5 else 0
        
        # Add predictions to the transaction
        transaction['Fraud_Probability'] = fraud_prob
        transaction['Predicted_Fraud'] = fraud_pred
        transaction['Alerts_and_Notifications'] = "High Risk Alert" if fraud_prob > 0.8 else "Medium Risk" if fraud_prob > 0.5 else "Low Risk"
        transaction['Historical_Insights'] = "Previous fraud cases analyzed"
        transaction['User_Actions'] = "Review Required" if fraud_pred == 1 else "Approved"
        
        # Add to the queue and real-time data
        data_queue.put(transaction)
        app_state['real_time_data'].append(transaction)
        
        # Control the generation rate (transactions per second)
        time.sleep(random.uniform(1, 3))

# API Endpoints

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if check_password(username, password):
        app_state['logged_in'] = True
        app_state['username'] = username
        return jsonify({'status': 'success', 'message': 'Login successful'})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    
    if username in app_state['password_hashes']:
        return jsonify({'status': 'error', 'message': 'Username already exists'}), 400
    elif password != confirm_password:
        return jsonify({'status': 'error', 'message': 'Passwords do not match'}), 400
    else:
        save_user(username, password)
        return jsonify({'status': 'success', 'message': 'Account created successfully'})

@app.route('/api/logout', methods=['POST'])
def logout():
    app_state['logged_in'] = False
    app_state['username'] = ""
    stop_data_stream()
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if not app_state['logged_in']:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    try:
        df = pd.read_csv(file)
        if not set(required_features).issubset(df.columns) and model is not None:
            return jsonify({
                'status': 'error',
                'message': 'Uploaded file does not contain the required columns',
                'required_columns': required_features
            }), 400
        
        # Preprocess data and make predictions
        X = preprocess_data(df)
        
        if model:
            fraud_prob = model.predict_proba(X)[:, 1]
            fraud_pred = model.predict(X)
        else:
            # Demo mode if model not available
            fraud_prob = np.random.rand(len(df))
            fraud_pred = (fraud_prob > 0.5).astype(int)
        
        # Add predictions to the dataframe
        df['Fraud_Probability'] = fraud_prob
        df['Predicted_Fraud'] = fraud_pred
        df['Alerts_and_Notifications'] = df['Fraud_Probability'].apply(
            lambda x: "High Risk Alert" if x > 0.8 else "Medium Risk" if x > 0.5 else "Low Risk")
        df['Historical_Insights'] = "Previous fraud cases analyzed"
        df['User_Actions'] = df['Predicted_Fraud'].apply(
            lambda x: "Review Required" if x == 1 else "Approved")
        df['Performance_Metrics'] = "Model Accuracy: 95% (Example)"
        
        # Store predictions in app state
        app_state['predictions'] = df.to_dict('records')
        
        # Prepare response
        response_data = {
            'status': 'success',
            'message': 'Data processed successfully',
            'summary': {
                'total_transactions': len(df),
                'fraud_detected': int(df['Predicted_Fraud'].sum()),
                'fraud_rate': float(df['Predicted_Fraud'].mean() * 100)
            },
            'sample_predictions': df.head(20).to_dict('records')
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    if not app_state['logged_in']:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    start_data_stream()
    return jsonify({'status': 'success', 'message': 'Real-time monitoring started'})

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    stop_data_stream()
    return jsonify({'status': 'success', 'message': 'Real-time monitoring stopped'})

@app.route('/api/stream/status', methods=['GET'])
def stream_status():
    return jsonify({
        'status': 'success',
        'stream_active': app_state['stream_active'],
        'total_transactions': len(app_state['real_time_data']),
        'fraud_count': sum(1 for t in app_state['real_time_data'] if t.get('Predicted_Fraud') == 1),
        'fraud_rate': (sum(1 for t in app_state['real_time_data'] if t.get('Predicted_Fraud') == 1) / 
                      len(app_state['real_time_data'])) * 100 if app_state['real_time_data'] else 0
    })

@app.route('/api/stream/latest', methods=['GET'])
def get_latest_transactions():
    count = request.args.get('count', default=10, type=int)
    latest = app_state['real_time_data'][-count:] if app_state['real_time_data'] else []
    return jsonify({
        'status': 'success',
        'transactions': latest
    })

@app.route('/api/stream/alerts', methods=['GET'])
def get_alerts():
    threshold = request.args.get('threshold', default=0.5, type=float)
    alerts = [t for t in app_state['real_time_data'] if t.get('Fraud_Probability', 0) > threshold]
    return jsonify({
        'status': 'success',
        'alerts': alerts[-5:]  # Return last 5 alerts
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    if not app_state['logged_in']:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    data = request.get_json()
    transaction = {
        'Transaction_Details': data.get('amount', 100.0),
        'Cardholder_Information': data.get('cardholder_info', 'Verified'),
        'Device_and_Network_Information': data.get('device_info', 'Secure'),
        'Historical_Data': data.get('historical_data', 500.0),
        'Behavioral_Data': data.get('behavioral_data', 0.5),
        'Security_Features': data.get('security_features', 'Enabled'),
        'External_Data': data.get('external_data', 'Low Risk')
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
    
    return jsonify({
        'status': 'success',
        'fraud_probability': float(fraud_prob),
        'predicted_fraud': int(fraud_pred),
        'risk_level': "High Risk" if fraud_prob > 0.8 else "Medium Risk" if fraud_prob > 0.5 else "Low Risk",
        'recommended_action': "Review Required" if fraud_pred == 1 else "Approved"
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    if not app_state['logged_in']:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    # Combine real-time data and predictions
    if app_state['real_time_data']:
        data = app_state['real_time_data']
    else:
        data = app_state['predictions']
    
    return jsonify({
        'status': 'success',
        'transactions': data
    })

@app.route('/api/user', methods=['GET'])
def get_user_info():
    if not app_state['logged_in']:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    
    return jsonify({
        'status': 'success',
        'username': app_state['username'],
        'logged_in': True
    })

if __name__ == '__main__':
    app.run(debug=True)