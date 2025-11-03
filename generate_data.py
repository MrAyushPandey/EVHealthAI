"""
EVHealthAI - Data Generation Script
Generates realistic synthetic EV sensor data for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_ev_data(n_vehicles=500, days_per_vehicle=200):
    """
    Generate synthetic EV sensor data with realistic degradation patterns
    """
    data = []
    
    for vehicle_id in range(1, n_vehicles + 1):
        # Vehicle characteristics
        vehicle_age_days = random.randint(0, 1095)  # 0-3 years
        battery_capacity = random.uniform(60, 100)  # kWh
        total_mileage = random.uniform(5000, 150000)  # km
        
        # Component initial health
        battery_health = random.uniform(85, 100)
        motor_health = random.uniform(85, 100)
        brake_health = random.uniform(80, 100)
        
        # Degradation rates (per day)
        battery_deg_rate = random.uniform(0.002, 0.008)
        motor_deg_rate = random.uniform(0.001, 0.005)
        brake_deg_rate = random.uniform(0.003, 0.01)
        
        start_date = datetime.now() - timedelta(days=vehicle_age_days)
        
        for day in range(days_per_vehicle):
            current_date = start_date + timedelta(days=day)
            
            # Daily usage
            daily_distance = random.uniform(20, 200)  # km per day
            total_mileage += daily_distance
            
            # Charge cycles
            charge_cycles = int(total_mileage / (battery_capacity * 5))
            
            # Temperature effects (seasonal)
            month = current_date.month
            if month in [12, 1, 2]:  # Winter
                temp_factor = random.uniform(0.8, 1.0)
                ambient_temp = random.uniform(-5, 10)
            elif month in [6, 7, 8]:  # Summer
                temp_factor = random.uniform(1.0, 1.3)
                ambient_temp = random.uniform(25, 40)
            else:  # Spring/Fall
                temp_factor = 1.0
                ambient_temp = random.uniform(10, 25)
            
            # Component degradation with realistic patterns
            battery_health -= battery_deg_rate * temp_factor
            motor_health -= motor_deg_rate * (daily_distance / 100)
            brake_health -= brake_deg_rate * (daily_distance / 100)
            
            # Add random fluctuations
            battery_health += random.uniform(-0.5, 0.2)
            motor_health += random.uniform(-0.3, 0.1)
            brake_health += random.uniform(-0.4, 0.1)
            
            # Ensure health doesn't go below 20 or above 100
            battery_health = max(20, min(100, battery_health))
            motor_health = max(20, min(100, motor_health))
            brake_health = max(20, min(100, brake_health))
            
            # Sensor readings
            battery_voltage = 400 * (battery_health / 100) + random.uniform(-5, 5)
            battery_temp = ambient_temp + random.uniform(5, 25) + (100 - battery_health) * 0.3
            battery_current = random.uniform(50, 200) if random.random() > 0.3 else 0
            
            motor_temp = ambient_temp + random.uniform(30, 60) + (100 - motor_health) * 0.5
            motor_rpm = random.uniform(1000, 6000) if daily_distance > 50 else random.uniform(0, 3000)
            motor_efficiency = motor_health * 0.95 + random.uniform(-2, 2)
            
            brake_temp = ambient_temp + random.uniform(10, 40)
            brake_pad_thickness = brake_health * 0.12 + random.uniform(-0.5, 0.5)  # mm
            brake_pressure = random.uniform(0, 100)
            
            # Anomalies (5% chance)
            if random.random() < 0.05:
                anomaly = 1
                battery_temp += random.uniform(10, 30)
                motor_temp += random.uniform(15, 35)
            else:
                anomaly = 0
            
            # Risk classification
            avg_health = (battery_health + motor_health + brake_health) / 3
            if avg_health >= 80:
                risk_level = 'Low'
            elif avg_health >= 60:
                risk_level = 'Medium'
            elif avg_health >= 40:
                risk_level = 'High'
            else:
                risk_level = 'Critical'
            
            # Maintenance needed
            maintenance_needed = 1 if avg_health < 50 or anomaly == 1 else 0
            
            # Estimated cost
            maintenance_cost = 0
            if battery_health < 60:
                maintenance_cost += random.uniform(500, 2000)
            if motor_health < 60:
                maintenance_cost += random.uniform(300, 1500)
            if brake_health < 60:
                maintenance_cost += random.uniform(100, 500)
            
            record = {
                'vehicle_id': f'EV_{vehicle_id:04d}',
                'timestamp': current_date.strftime('%Y-%m-%d'),
                'mileage': round(total_mileage, 2),
                'charge_cycles': charge_cycles,
                'ambient_temperature': round(ambient_temp, 2),
                'daily_distance': round(daily_distance, 2),
                
                # Battery features
                'battery_voltage': round(battery_voltage, 2),
                'battery_temperature': round(battery_temp, 2),
                'battery_current': round(battery_current, 2),
                'battery_health': round(battery_health, 2),
                
                # Motor features
                'motor_temperature': round(motor_temp, 2),
                'motor_rpm': round(motor_rpm, 2),
                'motor_efficiency': round(motor_efficiency, 2),
                'motor_health': round(motor_health, 2),
                
                # Brake features
                'brake_temperature': round(brake_temp, 2),
                'brake_pad_thickness': round(brake_pad_thickness, 2),
                'brake_pressure': round(brake_pressure, 2),
                'brake_health': round(brake_health, 2),
                
                # Target variables
                'overall_health': round(avg_health, 2),
                'risk_level': risk_level,
                'anomaly': anomaly,
                'maintenance_needed': maintenance_needed,
                'estimated_maintenance_cost': round(maintenance_cost, 2)
            }
            
            data.append(record)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("🚗 Generating EV Health Data...")
    print("=" * 60)
    
    # Generate data
    df = generate_ev_data(n_vehicles=500, days_per_vehicle=200)
    
    print(f"✅ Generated {len(df)} records for {df['vehicle_id'].nunique()} vehicles")
    print(f"✅ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\n📊 Data Statistics:")
    print(df.describe())
    
    # Save to CSV
    df.to_csv('data/ev_health_data.csv', index=False)
    print(f"\n💾 Data saved to: data/ev_health_data.csv")
    print(f"📁 File size: {len(df)} rows × {len(df.columns)} columns")
    
    # Display risk distribution
    print(f"\n🎯 Risk Level Distribution:")
    print(df['risk_level'].value_counts())
    
    print("\n✨ Data generation complete!")