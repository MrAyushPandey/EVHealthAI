"""
EVHealthAI - Flask API Backend
Serves ML models and predictions via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for models
models = {}
scalers = {}
encoders = {}
feature_names = []

def load_models():
    """Load all trained models"""
    global models, scalers, encoders, feature_names
    
    try:
        models['health_predictor'] = joblib.load('models/health_predictor.pkl')
        models['risk_classifier'] = joblib.load('models/risk_classifier.pkl')
        models['anomaly_detector'] = joblib.load('models/anomaly_detector.pkl')
        models['lstm_forecaster'] = keras.models.load_model('models/lstm_forecaster.h5')
        
        scalers = joblib.load('models/scalers.pkl')
        encoders = joblib.load('models/encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def prepare_features(data):
    """Prepare features from input data"""
    # Create DataFrame with all required features
    required_features = feature_names
    
    # Fill missing features with default values
    feature_dict = {}
    for feature in required_features:
        feature_dict[feature] = data.get(feature, 0)
    
    df = pd.DataFrame([feature_dict])
    return df[feature_names]

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'message': 'EVHealthAI API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects JSON with vehicle sensor data
    """
    try:
        data = request.json
        
        # Prepare features
        X = prepare_features(data)
        
        # Health Score Prediction
        X_health_scaled = scalers['health_scaler'].transform(X)
        health_score = float(models['health_predictor'].predict(X_health_scaled)[0])
        
        # Risk Classification
        X_risk_scaled = scalers['risk_scaler'].transform(X)
        risk_encoded = models['risk_classifier'].predict(X_risk_scaled)[0]
        risk_level = encoders['risk_encoder'].inverse_transform([risk_encoded])[0]
        risk_proba = models['risk_classifier'].predict_proba(X_risk_scaled)[0]
        
        # Anomaly Detection
        X_anomaly_scaled = scalers['anomaly_scaler'].transform(X)
        anomaly_pred = models['anomaly_detector'].predict(X_anomaly_scaled)[0]
        is_anomaly = bool(anomaly_pred == -1)
        anomaly_score = float(models['anomaly_detector'].score_samples(X_anomaly_scaled)[0])
        
        # Component Health (from input or calculated)
        battery_health = data.get('battery_health', health_score)
        motor_health = data.get('motor_health', health_score)
        brake_health = data.get('brake_health', health_score)
        
        # Generate recommendations
        recommendations = generate_recommendations(
            battery_health, motor_health, brake_health, is_anomaly
        )
        
        # Maintenance cost estimation
        cost = estimate_maintenance_cost(battery_health, motor_health, brake_health)
        
        response = {
            'success': True,
            'vehicle_id': data.get('vehicle_id', 'Unknown'),
            'prediction': {
                'overall_health': round(health_score, 2),
                'risk_level': risk_level,
                'risk_probabilities': {
                    'low': round(float(risk_proba[0]) * 100, 2),
                    'medium': round(float(risk_proba[1]) * 100, 2) if len(risk_proba) > 1 else 0,
                    'high': round(float(risk_proba[2]) * 100, 2) if len(risk_proba) > 2 else 0,
                    'critical': round(float(risk_proba[3]) * 100, 2) if len(risk_proba) > 3 else 0
                },
                'is_anomaly': is_anomaly,
                'anomaly_score': round(anomaly_score, 4)
            },
            'components': {
                'battery': {
                    'health': round(battery_health, 2),
                    'status': get_status(battery_health)
                },
                'motor': {
                    'health': round(motor_health, 2),
                    'status': get_status(motor_health)
                },
                'brakes': {
                    'health': round(brake_health, 2),
                    'status': get_status(brake_health)
                }
            },
            'maintenance': {
                'needed': health_score < 60 or is_anomaly,
                'priority': get_priority(health_score, is_anomaly),
                'estimated_cost': cost,
                'recommendations': recommendations
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint - responds to natural language queries
    """
    try:
        data = request.json
        message = data.get('message', '').lower()
        vehicle_data = data.get('vehicle_data', {})
        
        # Simple rule-based responses (can be enhanced with NLP)
        if 'health' in message or 'status' in message:
            # Get prediction
            pred_response = predict()
            pred_data = pred_response.get_json()
            
            if pred_data['success']:
                health = pred_data['prediction']['overall_health']
                response_text = f"Your EV's overall health is {health}%. "
                
                if health >= 80:
                    response_text += "Great condition! Keep up with regular maintenance."
                elif health >= 60:
                    response_text += "Good condition, but schedule a check-up soon."
                elif health >= 40:
                    response_text += "⚠️ Attention needed! Some components require maintenance."
                else:
                    response_text += "🚨 Critical! Immediate inspection recommended."
            else:
                response_text = "I need vehicle sensor data to check health status."
        
        elif 'battery' in message:
            response_text = "I can analyze your battery health. Please provide: battery voltage, temperature, and current readings."
        
        elif 'motor' in message:
            response_text = "For motor analysis, I need: motor temperature, RPM, and efficiency data."
        
        elif 'brake' in message or 'brakes' in message:
            response_text = "To check brakes, provide: brake temperature, pad thickness, and brake pressure."
        
        elif 'cost' in message or 'price' in message:
            response_text = "Maintenance costs vary by component:\n• Battery: $800-$2000\n• Motor: $500-$1500\n• Brakes: $200-$500"
        
        elif 'predict' in message or 'forecast' in message:
            response_text = "I can forecast your vehicle's health for the next 6 months using historical data patterns."
        
        elif 'anomaly' in message or 'problem' in message or 'issue' in message:
            response_text = "I'll scan for anomalies in your sensor data. Unusual patterns may indicate developing issues."
        
        elif 'help' in message or 'what can you do' in message:
            response_text = """I can help with:
            
🔍 Health Analysis - Overall vehicle health assessment
🔋 Battery Status - Battery degradation and capacity
⚙️ Motor Performance - Motor efficiency and temperature
🛑 Brake Condition - Brake pad wear and system health
⚠️ Anomaly Detection - Identify unusual patterns
💰 Cost Estimation - Maintenance cost predictions
📊 Health Forecast - Future health predictions

Just ask me anything about your EV!"""
        
        else:
            response_text = "I'm EVHealthAI, your EV maintenance assistant! Ask me about your vehicle's health, battery, motor, brakes, or maintenance needs."
        
        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/dashboard-data', methods=['POST'])
def get_dashboard_data():
    """
    Get comprehensive dashboard data
    """
    try:
        data = request.json
        
        # Get prediction
        X = prepare_features(data)
        
        # All predictions
        X_health_scaled = scalers['health_scaler'].transform(X)
        health_score = float(models['health_predictor'].predict(X_health_scaled)[0])
        
        X_risk_scaled = scalers['risk_scaler'].transform(X)
        risk_encoded = models['risk_classifier'].predict(X_risk_scaled)[0]
        risk_level = encoders['risk_encoder'].inverse_transform([risk_encoded])[0]
        
        # Component data
        components = {
            'battery': data.get('battery_health', health_score),
            'motor': data.get('motor_health', health_score),
            'brakes': data.get('brake_health', health_score)
        }
        
        # Historical trend simulation (last 30 days)
        trend_data = []
        base_health = health_score
        for i in range(30, 0, -1):
            day_health = base_health + np.random.uniform(-5, 2)
            day_health = max(20, min(100, day_health))
            trend_data.append({
                'day': i,
                'health': round(day_health, 2)
            })
            base_health = day_health
        
        # Forecast (next 30 days)
        forecast_data = []
        degradation_rate = 0.1
        current = health_score
        for i in range(1, 31):
            current -= degradation_rate + np.random.uniform(-0.2, 0.1)
            current = max(20, current)
            forecast_data.append({
                'day': i,
                'health': round(current, 2)
            })
        
        response = {
            'success': True,
            'summary': {
                'overall_health': round(health_score, 2),
                'risk_level': risk_level,
                'components': components
            },
            'trends': {
                'historical': trend_data,
                'forecast': forecast_data
            },
            'alerts': generate_alerts(components, health_score),
            'recommendations': generate_recommendations(
                components['battery'],
                components['motor'],
                components['brakes'],
                False
            )
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_status(health):
    """Get status label based on health score"""
    if health >= 80:
        return 'Excellent'
    elif health >= 60:
        return 'Good'
    elif health >= 40:
        return 'Fair'
    else:
        return 'Poor'

def get_priority(health, is_anomaly):
    """Get maintenance priority"""
    if is_anomaly or health < 40:
        return 'Critical'
    elif health < 60:
        return 'High'
    elif health < 80:
        return 'Medium'
    else:
        return 'Low'

def estimate_maintenance_cost(battery, motor, brakes):
    """Estimate maintenance cost"""
    cost = 0
    if battery < 60:
        cost += np.random.uniform(800, 2000)
    if motor < 60:
        cost += np.random.uniform(500, 1500)
    if brakes < 60:
        cost += np.random.uniform(200, 500)
    
    return f"${int(cost)}-${int(cost * 1.3)}"

def generate_recommendations(battery, motor, brakes, is_anomaly):
    """Generate maintenance recommendations"""
    recommendations = []
    
    if is_anomaly:
        recommendations.append({
            'component': 'System',
            'priority': 'Critical',
            'action': 'Immediate diagnostic scan - unusual patterns detected',
            'icon': '🚨'
        })
    
    if battery < 60:
        recommendations.append({
            'component': 'Battery',
            'priority': 'High',
            'action': 'Schedule battery inspection and capacity test',
            'icon': '🔋'
        })
    
    if motor < 60:
        recommendations.append({
            'component': 'Motor',
            'priority': 'High',
            'action': 'Motor diagnostics and cooling system check',
            'icon': '⚙️'
        })
    
    if brakes < 60:
        recommendations.append({
            'component': 'Brakes',
            'priority': 'Medium',
            'action': 'Brake pad replacement and system inspection',
            'icon': '🛑'
        })
    
    if not recommendations:
        recommendations.append({
            'component': 'All Systems',
            'priority': 'Low',
            'action': 'Continue regular maintenance schedule',
            'icon': '✅'
        })
    
    return recommendations

def generate_alerts(components, overall_health):
    """Generate alert messages"""
    alerts = []
    
    if overall_health < 40:
        alerts.append({
            'level': 'critical',
            'message': 'Critical health level - immediate attention required',
            'icon': '🚨'
        })
    
    for name, health in components.items():
        if health < 50:
            alerts.append({
                'level': 'warning',
                'message': f'{name.capitalize()} health is low ({health:.1f}%)',
                'icon': '⚠️'
            })
    
    return alerts

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗⚡ EVHealthAI API Server")
    print("="*60)
    
    # Load models
    if load_models():
        print("\n✅ Starting Flask server...")
        print("📡 API available at: http://localhost:5000")
        print("\nEndpoints:")
        print("  GET  /api/health          - Health check")
        print("  POST /api/predict         - Vehicle prediction")
        print("  POST /api/chat            - Chatbot interaction")
        print("  POST /api/dashboard-data  - Dashboard data")
        print("\n" + "="*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please train models first:")
        print("   python train_models.py")