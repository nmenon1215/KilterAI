#!/usr/bin/env python3
"""
Kilter Board Grade Prediction Script

This script uses the trained models to predict the grade of a Kilter Board route
based on its features.

Usage:
    python predict_grade.py --input climb_features.json --model [nn|rf|ensemble]
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Feature names used by the models
FEATURE_NAMES = ['is_listed', 'is_benchmark', 'num_holds', 'num_start_holds', 'num_finish_holds', 'grid_0_0', 'grid_0_1', 'grid_0_2', 'grid_0_3', 'grid_0_4', 'grid_0_5', 'grid_0_6', 'grid_0_7', 'grid_0_8', 'grid_0_9', 'grid_0_10', 'grid_1_0', 'grid_1_1', 'grid_1_2', 'grid_1_3', 'grid_1_4', 'grid_1_5', 'grid_1_6', 'grid_1_7', 'grid_1_8', 'grid_1_9', 'grid_1_10', 'grid_2_1', 'grid_2_2', 'grid_2_3', 'grid_2_4', 'grid_2_5', 'grid_2_6', 'grid_2_7', 'grid_2_8', 'grid_2_9', 'grid_2_10', 'grid_2_11', 'grid_3_1', 'grid_3_2', 'grid_3_3', 'grid_3_4', 'grid_3_5', 'grid_3_6', 'grid_3_7', 'grid_3_8', 'grid_3_9', 'grid_3_10', 'grid_3_11', 'grid_4_1', 'grid_4_2', 'grid_4_3', 'grid_4_4', 'grid_4_5', 'grid_4_6', 'grid_4_7', 'grid_4_8', 'grid_4_9', 'grid_4_10', 'grid_4_11', 'grid_5_0', 'grid_5_1', 'grid_5_2', 'grid_5_3', 'grid_5_4', 'grid_5_5', 'grid_5_6', 'grid_5_7', 'grid_5_8', 'grid_5_9', 'grid_5_10', 'grid_5_11', 'grid_6_0', 'grid_6_1', 'grid_6_2', 'grid_6_3', 'grid_6_4', 'grid_6_5', 'grid_6_6', 'grid_6_7', 'grid_6_8', 'grid_6_9', 'grid_6_11', 'grid_7_2', 'grid_7_3', 'grid_7_4', 'grid_7_5', 'grid_7_6', 'grid_7_7', 'grid_7_8', 'grid_7_11', 'grid_8_2', 'grid_8_3', 'grid_8_4', 'grid_8_5', 'grid_8_6', 'grid_8_7', 'grid_8_8', 'grid_8_10', 'grid_8_11', 'grid_9_2', 'grid_9_3', 'grid_9_4', 'grid_9_5', 'grid_9_6', 'grid_9_7', 'grid_9_8', 'grid_10_2', 'grid_10_3', 'grid_10_4', 'grid_10_5', 'grid_10_6', 'grid_10_7', 'grid_10_8', 'grid_10_9', 'grid_11_2', 'grid_11_3', 'grid_11_4', 'grid_11_5', 'grid_11_6', 'grid_11_7', 'grid_11_8', 'density_bottom', 'density_top', 'density_left', 'density_right', 'avg_distance', 'max_distance', 'min_distance', 'std_distance']

class GradeNN(nn.Module):
    """Neural network model for grade prediction."""
    
    def __init__(self, input_size, hidden_sizes):
        super(GradeNN, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def load_models(model_dir):
    """Load the trained models."""
    models = {}
    
    # Load neural network model
    nn_path = os.path.join(model_dir, 'nn_model.pt')
    if os.path.exists(nn_path):
        # Load architecture
        with open(os.path.join(model_dir, 'nn_architecture.json'), 'r') as f:
            architecture = json.load(f)
        
        # Initialize model
        nn_model = GradeNN(architecture['input_size'], architecture['hidden_sizes'])
        
        # Load weights
        nn_model.load_state_dict(torch.load(nn_path, map_location=torch.device('cpu')))
        nn_model.eval()
        
        models['nn'] = nn_model
    
    # Load Random Forest model
    rf_path = os.path.join(model_dir, 'rf_model.joblib')
    if os.path.exists(rf_path):
        models['rf'] = joblib.load(rf_path)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        models['scaler'] = joblib.load(scaler_path)
    
    return models

def grade_to_string(grade_value):
    """Convert numerical grade value to string representation."""
    grade_int = int(round(grade_value))
    return f"V{grade_int}"

def extract_features_from_json(json_file, feature_names):
    """Extract features from a JSON file with climb data."""
    with open(json_file, 'r') as f:
        climb_data = json.load(f)
    
    # Initialize feature vector with zeros
    features = {}
    for feature in feature_names:
        features[feature] = 0
    
    # Extract features that are directly in the climb data
    for feature in feature_names:
        if feature in climb_data:
            features[feature] = climb_data[feature]
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all required features are present
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            df[feature] = 0
    
    return df[feature_names]

def predict_grade(features, models, model_type='ensemble'):
    """
    Predict the grade of a climb based on its features.
    
    Args:
        features: DataFrame with climb features
        models: Dictionary of loaded models
        model_type: Type of model to use ('nn', 'rf', or 'ensemble')
        
    Returns:
        Dictionary with prediction results
    """
    # Scale features
    X_scaled = models['scaler'].transform(features)
    
    predictions = {}
    
    # Neural Network prediction
    if 'nn' in models and (model_type == 'nn' or model_type == 'ensemble'):
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Make prediction
        with torch.no_grad():
            nn_pred = models['nn'](X_tensor).item()
        
        predictions['nn'] = nn_pred
    
    # Random Forest prediction
    if 'rf' in models and (model_type == 'rf' or model_type == 'ensemble'):
        # Make prediction
        rf_pred = models['rf'].predict(X_scaled)[0]
        
        predictions['rf'] = rf_pred
    
    # Calculate final prediction based on model type
    if model_type == 'ensemble' and len(predictions) == 2:
        # Average predictions from both models
        final_prediction = (predictions['nn'] + predictions['rf']) / 2
    elif model_type == 'nn' and 'nn' in predictions:
        final_prediction = predictions['nn']
    elif model_type == 'rf' and 'rf' in predictions:
        final_prediction = predictions['rf']
    else:
        # Fallback to any available prediction
        final_prediction = next(iter(predictions.values()))
    
    # Convert to V-grade
    v_grade = grade_to_string(final_prediction)
    
    return {
        'grade_value': final_prediction,
        'grade': v_grade,
        'model_predictions': predictions
    }

def main():
    parser = argparse.ArgumentParser(description='Predict climbing grade')
    parser.add_argument('--input', type=str, required=True, help='Path to climb features JSON file')
    parser.add_argument('--model', type=str, default='ensemble', choices=['nn', 'rf', 'ensemble'],
                        help='Model to use for prediction')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory with trained models')
    
    args = parser.parse_args()
    
    # Load models
    models = load_models(args.model_dir)
    
    if not models:
        print("Error: No models found in {args.model_dir}")
        return
    
    # Extract features
    features = extract_features_from_json(args.input, FEATURE_NAMES)
    
    # Make prediction
    prediction = predict_grade(features, models, args.model)
    
    # Print results
    print(f"\nPrediction for climb: {args.input}\n")
    print(f"Predicted grade: {prediction['grade']} ({prediction['grade_value']:.2f})")
    
    if 'model_predictions' in prediction:
        print("\nIndividual model predictions:")
        for model, pred_value in prediction['model_predictions'].items():
            print(f"- {model.upper()}: {grade_to_string(pred_value)} ({pred_value:.2f})")

if __name__ == "__main__":
    main()
