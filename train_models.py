"""
EVHealthAI - Model Training Script
Trains multiple ML models for EV health prediction
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import xgboost as xgb

# Deep Learning
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class EVHealthPredictor:
    """
    Complete ML Pipeline for EV Health Prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.metrics = {}
        
    def load_and_preprocess_data(self, filepath='data/ev_health_data.csv'):
        """Load and preprocess the data"""
        print("📂 Loading data...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
        
        # Sort by vehicle and date
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        # Create rolling features for time-series patterns
        for col in ['battery_health', 'motor_health', 'brake_health']:
            df[f'{col}_rolling_7d'] = df.groupby('vehicle_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{col}_rolling_30d'] = df.groupby('vehicle_id')[col].transform(
                lambda x: x.rolling(30, min_periods=1).mean()
            )
        
        # Degradation rate features
        df['battery_degradation_rate'] = df.groupby('vehicle_id')['battery_health'].transform(
            lambda x: x.diff().fillna(0)
        )
        df['motor_degradation_rate'] = df.groupby('vehicle_id')['motor_health'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        print(f"✅ Loaded {len(df)} records")
        print(f"✅ Features: {len(df.columns)} columns")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        
        # Feature columns (exclude targets and IDs)
        exclude_cols = ['vehicle_id', 'timestamp', 'overall_health', 'risk_level', 
                       'maintenance_needed', 'estimated_maintenance_cost', 'anomaly',
                       'battery_health', 'motor_health', 'brake_health']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        
        # Target variables
        y_health = df['overall_health']
        y_risk = df['risk_level']
        y_anomaly = df['anomaly']
        y_maintenance = df['maintenance_needed']
        y_cost = df['estimated_maintenance_cost']
        
        # Encode risk levels
        self.encoders['risk_encoder'] = LabelEncoder()
        y_risk_encoded = self.encoders['risk_encoder'].fit_transform(y_risk)
        
        print(f"\n📊 Feature Matrix: {X.shape}")
        print(f"📊 Target - Health Score: {y_health.shape}")
        print(f"📊 Target - Risk Level: {y_risk.value_counts().to_dict()}")
        
        return X, y_health, y_risk_encoded, y_anomaly, y_maintenance, y_cost
    
    def train_health_score_model(self, X_train, X_test, y_train, y_test):
        """Train Random Forest for health score prediction"""
        print("\n" + "="*60)
        print("🎯 Training Model 1: Health Score Predictor (Random Forest)")
        print("="*60)
        
        # Scale features
        self.scalers['health_scaler'] = StandardScaler()
        X_train_scaled = self.scalers['health_scaler'].fit_transform(X_train)
        X_test_scaled = self.scalers['health_scaler'].transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['health_predictor'] = rf_model
        self.metrics['health_predictor'] = {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'R2_Score': round(r2, 4)
        }
        
        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ R² Score: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features:")
        print(feature_importance.head(10))
        
        return rf_model, feature_importance
    
    def train_risk_classifier(self, X_train, X_test, y_train, y_test):
        """Train XGBoost for risk classification"""
        print("\n" + "="*60)
        print("🎯 Training Model 2: Risk Level Classifier (XGBoost)")
        print("="*60)
        
        # Scale features
        self.scalers['risk_scaler'] = StandardScaler()
        X_train_scaled = self.scalers['risk_scaler'].fit_transform(X_train)
        X_test_scaled = self.scalers['risk_scaler'].transform(X_test)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        xgb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.models['risk_classifier'] = xgb_model
        self.metrics['risk_classifier'] = {
            'Accuracy': round(accuracy, 4),
            'F1_Score': round(f1, 4)
        }
        
        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ F1 Score: {f1:.4f}")
        
        print(f"\n📊 Classification Report:")
        risk_labels = self.encoders['risk_encoder'].classes_
        print(classification_report(y_test, y_pred, target_names=risk_labels))
        
        return xgb_model
    
    def train_anomaly_detector(self, X_train):
        """Train Isolation Forest for anomaly detection"""
        print("\n" + "="*60)
        print("🎯 Training Model 3: Anomaly Detector (Isolation Forest)")
        print("="*60)
        
        # Scale features
        self.scalers['anomaly_scaler'] = StandardScaler()
        X_train_scaled = self.scalers['anomaly_scaler'].fit_transform(X_train)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        iso_forest.fit(X_train_scaled)
        
        # Predictions (-1 for anomaly, 1 for normal)
        anomaly_pred = iso_forest.predict(X_train_scaled)
        anomaly_pred = np.where(anomaly_pred == -1, 1, 0)
        
        detected = np.sum(anomaly_pred)
        detection_rate = (detected / len(anomaly_pred)) * 100
        
        self.models['anomaly_detector'] = iso_forest
        self.metrics['anomaly_detector'] = {
            'Detection_Rate': f"{detection_rate:.2f}%",
            'Anomalies_Found': int(detected)
        }
        
        print(f"✅ Anomalies Detected: {detected} ({detection_rate:.2f}%)")
        
        return iso_forest
    
    def train_lstm_forecaster(self, df, sequence_length=30):
        """Train LSTM for time-series forecasting"""
        print("\n" + "="*60)
        print("🎯 Training Model 4: Health Forecaster (LSTM)")
        print("="*60)
        
        # Prepare sequences for LSTM
        sequences = []
        targets = []
        
        for vehicle in df['vehicle_id'].unique():
            vehicle_data = df[df['vehicle_id'] == vehicle].sort_values('timestamp')
            health_values = vehicle_data['overall_health'].values
            
            for i in range(len(health_values) - sequence_length):
                seq = health_values[i:i+sequence_length]
                target = health_values[i+sequence_length]
                sequences.append(seq)
                targets.append(target)
        
        X_seq = np.array(sequences).reshape(-1, sequence_length, 1)
        y_seq = np.array(targets)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        lstm_model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("🔄 Training LSTM (this may take a few minutes)...")
        history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred = lstm_model.predict(X_test, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        self.models['lstm_forecaster'] = lstm_model
        self.metrics['lstm_forecaster'] = {
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4)
        }
        
        print(f"✅ LSTM RMSE: {rmse:.4f}")
        print(f"✅ LSTM MAE: {mae:.4f}")
        
        return lstm_model
    
    def save_models(self):
        """Save all models and scalers"""
        print("\n" + "="*60)
        print("💾 Saving Models...")
        print("="*60)
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'lstm_forecaster':
                joblib.dump(model, f'models/{name}.pkl')
                print(f"✅ Saved: models/{name}.pkl")
        
        # Save LSTM separately
        if 'lstm_forecaster' in self.models:
            self.models['lstm_forecaster'].save('models/lstm_forecaster.h5')
            print(f"✅ Saved: models/lstm_forecaster.h5")
        
        # Save scalers and encoders
        joblib.dump(self.scalers, 'models/scalers.pkl')
        joblib.dump(self.encoders, 'models/encoders.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        print(f"✅ Saved: Scalers and Encoders")
        
        # Save metrics
        with open('models/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"✅ Saved: models/metrics.json")
        
        print("\n✨ All models saved successfully!")

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("🚗⚡ EVHealthAI - Model Training Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize predictor
    predictor = EVHealthPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    
    # Prepare features
    X, y_health, y_risk, y_anomaly, y_maintenance, y_cost = predictor.prepare_features(df)
    
    # Train-test split
    X_train, X_test, y_health_train, y_health_test = train_test_split(
        X, y_health, test_size=0.2, random_state=42
    )
    _, _, y_risk_train, y_risk_test = train_test_split(
        X, y_risk, test_size=0.2, random_state=42
    )
    
    # Train all models
    predictor.train_health_score_model(X_train, X_test, y_health_train, y_health_test)
    predictor.train_risk_classifier(X_train, X_test, y_risk_train, y_risk_test)
    predictor.train_anomaly_detector(X_train)
    predictor.train_lstm_forecaster(df)
    
    # Save models
    predictor.save_models()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    for model_name, metrics in predictor.metrics.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\n" + "="*60)
    print(f"✨ Training Complete! - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()