"""
EVHealthAI - Prediction and Visualization Script
Makes predictions and creates interactive dashboards
"""

import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class EVHealthDashboard:
    """
    Create interactive dashboards for EV health monitoring
    """
    
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        print("📂 Loading trained models...")
        
        self.models = {
            'health_predictor': joblib.load('models/health_predictor.pkl'),
            'risk_classifier': joblib.load('models/risk_classifier.pkl'),
            'anomaly_detector': joblib.load('models/anomaly_detector.pkl'),
            'lstm_forecaster': keras.models.load_model('models/lstm_forecaster.h5')
        }
        
        self.scalers = joblib.load('models/scalers.pkl')
        self.encoders = joblib.load('models/encoders.pkl')
        self.feature_names = joblib.load('models/feature_names.pkl')
        
        with open('models/metrics.json', 'r') as f:
            self.metrics = json.load(f)
        
        print("✅ Models loaded successfully!")
    
    def predict_single_vehicle(self, vehicle_data):
        """Make predictions for a single vehicle"""
        
        # Prepare features
        X = vehicle_data[self.feature_names].values
        
        # Health Score Prediction
        X_health_scaled = self.scalers['health_scaler'].transform(X)
        health_score = self.models['health_predictor'].predict(X_health_scaled)[0]
        
        # Risk Classification
        X_risk_scaled = self.scalers['risk_scaler'].transform(X)
        risk_encoded = self.models['risk_classifier'].predict(X_risk_scaled)[0]
        risk_level = self.encoders['risk_encoder'].inverse_transform([risk_encoded])[0]
        
        # Anomaly Detection
        X_anomaly_scaled = self.scalers['anomaly_scaler'].transform(X)
        anomaly_pred = self.models['anomaly_detector'].predict(X_anomaly_scaled)[0]
        is_anomaly = "Yes" if anomaly_pred == -1 else "No"
        
        results = {
            'health_score': round(health_score, 2),
            'risk_level': risk_level,
            'is_anomaly': is_anomaly,
            'battery_health': vehicle_data['battery_health'].values[0],
            'motor_health': vehicle_data['motor_health'].values[0],
            'brake_health': vehicle_data['brake_health'].values[0]
        }
        
        return results
    
    def create_health_gauge(self, health_score, title):
        """Create health gauge chart"""
        
        # Color based on health
        if health_score >= 80:
            color = "green"
        elif health_score >= 60:
            color = "yellow"
        elif health_score >= 40:
            color = "orange"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 24}},
            delta = {'reference': 80, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 60], 'color': '#ffe6cc'},
                    {'range': [60, 80], 'color': '#fff9cc'},
                    {'range': [80, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    
    def create_component_comparison(self, results):
        """Create component health comparison chart"""
        
        components = ['Battery', 'Motor', 'Brakes']
        health_values = [
            results['battery_health'],
            results['motor_health'],
            results['brake_health']
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=health_values,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in health_values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Component Health Comparison",
            xaxis_title="Components",
            yaxis_title="Health Score (%)",
            yaxis=dict(range=[0, 110]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_time_series_plot(self, df, vehicle_id):
        """Create time-series plot for vehicle health"""
        
        vehicle_df = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=vehicle_df['timestamp'],
            y=vehicle_df['battery_health'],
            mode='lines',
            name='Battery Health',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=vehicle_df['timestamp'],
            y=vehicle_df['motor_health'],
            mode='lines',
            name='Motor Health',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=vehicle_df['timestamp'],
            y=vehicle_df['brake_health'],
            mode='lines',
            name='Brake Health',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f"Health Trends for {vehicle_id}",
            xaxis_title="Date",
            yaxis_title="Health Score (%)",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_risk_distribution(self, df):
        """Create risk level distribution pie chart"""
        
        risk_counts = df['risk_level'].value_counts()
        
        colors = {
            'Low': '#2ecc71',
            'Medium': '#f39c12',
            'High': '#e67e22',
            'Critical': '#e74c3c'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[colors[risk] for risk in risk_counts.index]),
            hole=0.4
        )])
        
        fig.update_layout(
            title="Overall Risk Distribution",
            height=400
        )
        
        return fig
    
    def create_model_performance_chart(self):
        """Create model performance comparison"""
        
        models = list(self.metrics.keys())
        model_labels = [m.replace('_', ' ').title() for m in models]
        
        # Extract primary metric for each model
        primary_metrics = []
        metric_names = []
        
        for model in models:
            metrics = self.metrics[model]
            if 'R2_Score' in metrics:
                primary_metrics.append(metrics['R2_Score'] * 100)
                metric_names.append('R² Score (%)')
            elif 'Accuracy' in metrics:
                primary_metrics.append(metrics['Accuracy'] * 100)
                metric_names.append('Accuracy (%)')
            elif 'F1_Score' in metrics:
                primary_metrics.append(metrics['F1_Score'] * 100)
                metric_names.append('F1 Score (%)')
            else:
                primary_metrics.append(0)
                metric_names.append('N/A')
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_labels,
                y=primary_metrics,
                marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                text=[f"{v:.2f}%" for v in primary_metrics],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Performance Score (%)",
            yaxis=dict(range=[0, 110]),
            height=400
        )
        
        return fig
    
    def create_maintenance_forecast(self, df, vehicle_id, days_ahead=180):
        """Forecast maintenance needs for next 6 months"""
        
        vehicle_df = df[df['vehicle_id'] == vehicle_id].sort_values('timestamp').tail(30)
        current_health = vehicle_df['overall_health'].values
        
        # Simple linear projection (can be enhanced with LSTM)
        recent_trend = np.polyfit(range(len(current_health)), current_health, 1)
        
        future_days = range(len(current_health), len(current_health) + days_ahead)
        forecast = np.polyval(recent_trend, future_days)
        forecast = np.clip(forecast, 0, 100)
        
        future_dates = pd.date_range(
            start=vehicle_df['timestamp'].max(),
            periods=days_ahead,
            freq='D'
        )
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=vehicle_df['timestamp'],
            y=current_health,
            mode='lines',
            name='Historical Health',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode='lines',
            name='Forecasted Health',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Maintenance threshold
        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="orange",
            annotation_text="Maintenance Threshold"
        )
        
        fig.update_layout(
            title=f"6-Month Health Forecast for {vehicle_id}",
            xaxis_title="Date",
            yaxis_title="Health Score (%)",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def generate_dashboard(self, df):
        """Generate complete interactive dashboard"""
        
        print("\n" + "="*60)
        print("📊 Generating Interactive Dashboard...")
        print("="*60)
        
        # Select a sample vehicle for detailed analysis
        sample_vehicle = df['vehicle_id'].iloc[0]
        vehicle_data = df[df['vehicle_id'] == sample_vehicle].tail(1)
        
        # Make predictions
        results = self.predict_single_vehicle(vehicle_data)
        
        # Create all visualizations
        print("📈 Creating visualizations...")
        
        # 1. Health Gauges
        overall_gauge = self.create_health_gauge(results['health_score'], "Overall Health")
        battery_gauge = self.create_health_gauge(results['battery_health'], "Battery Health")
        motor_gauge = self.create_health_gauge(results['motor_health'], "Motor Health")
        brake_gauge = self.create_health_gauge(results['brake_health'], "Brake Health")
        
        # 2. Component Comparison
        component_chart = self.create_component_comparison(results)
        
        # 3. Time Series
        time_series = self.create_time_series_plot(df, sample_vehicle)
        
        # 4. Risk Distribution
        risk_chart = self.create_risk_distribution(df)
        
        # 5. Model Performance
        performance_chart = self.create_model_performance_chart()
        
        # 6. Maintenance Forecast
        forecast_chart = self.create_maintenance_forecast(df, sample_vehicle)
        
        # Save individual charts
        overall_gauge.write_html('visualizations/overall_health_gauge.html')
        component_chart.write_html('visualizations/component_comparison.html')
        time_series.write_html('visualizations/health_trends.html')
        risk_chart.write_html('visualizations/risk_distribution.html')
        performance_chart.write_html('visualizations/model_performance.html')
        forecast_chart.write_html('visualizations/maintenance_forecast.html')
        
        print("✅ All visualizations saved to 'visualizations/' folder")
        
        # Create summary report
        summary = {
            'vehicle_id': sample_vehicle,
            'prediction_results': results,
            'risk_assessment': {
                'level': results['risk_level'],
                'anomaly_detected': results['is_anomaly']
            },
            'recommendations': self.generate_recommendations(results)
        }
        
        with open('visualizations/prediction_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("✅ Prediction summary saved")
        
        return summary
    
    def generate_recommendations(self, results):
        """Generate maintenance recommendations"""
        
        recommendations = []
        
        if results['battery_health'] < 60:
            recommendations.append({
                'component': 'Battery',
                'priority': 'High',
                'action': 'Schedule battery inspection and possible replacement',
                'estimated_cost': '$800-$2000'
            })
        
        if results['motor_health'] < 60:
            recommendations.append({
                'component': 'Motor',
                'priority': 'High',
                'action': 'Motor diagnostics and cooling system check',
                'estimated_cost': '$500-$1500'
            })
        
        if results['brake_health'] < 60:
            recommendations.append({
                'component': 'Brakes',
                'priority': 'Medium',
                'action': 'Brake pad replacement and system inspection',
                'estimated_cost': '$200-$500'
            })
        
        if results['is_anomaly'] == 'Yes':
            recommendations.append({
                'component': 'General',
                'priority': 'Critical',
                'action': 'Immediate diagnostic scan - unusual patterns detected',
                'estimated_cost': '$100-$300'
            })
        
        if not recommendations:
            recommendations.append({
                'component': 'All Systems',
                'priority': 'Low',
                'action': 'Continue regular maintenance schedule',
                'estimated_cost': '$0'
            })
        
        return recommendations

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("🚗⚡ EVHealthAI - Prediction & Visualization")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv('data/ev_health_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize dashboard
    dashboard = EVHealthDashboard()
    
    # Generate dashboard
    summary = dashboard.generate_dashboard(df)
    
    # Print summary
    print("\n" + "="*60)
    print("📋 PREDICTION SUMMARY")
    print("="*60)
    print(f"\nVehicle ID: {summary['vehicle_id']}")
    print(f"Overall Health Score: {summary['prediction_results']['health_score']}")
    print(f"Risk Level: {summary['prediction_results']['risk_level']}")
    print(f"Anomaly Detected: {summary['prediction_results']['is_anomaly']}")
    
    print("\n🔧 Maintenance Recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"\n  {i}. {rec['component']} - Priority: {rec['priority']}")
        print(f"     Action: {rec['action']}")
        print(f"     Cost: {rec['estimated_cost']}")
    
    print("\n" + "="*60)
    print("✨ Dashboard generation complete!")
    print("📁 Check 'visualizations/' folder for all charts")
    print("="*60)

if __name__ == "__main__":
    main()