"""
Kilter Board Grade Prediction Model

This script implements and evaluates machine learning models for predicting
the grade of Kilter Board climbing routes based on hold positions and other features.

It implements both a neural network (primary model) and a Random Forest (secondary model)
as specified in the project requirements.

Usage:
    python train_kilter_model.py --data_path dataset/ml_features.csv --output_dir models/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the data for training.

    Args:
        data_path: Path to the ML features CSV file

    Returns:
        X: Feature matrix
        y: Target variable (grade value)
        feature_names: Names of features
        grade_labels: Original grade labels
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # Check for grade_value column
    if 'grade_value' not in df.columns:
        print("Error: 'grade_value' column not found in the dataset.")
        if 'grade' in df.columns:
            print("Creating grade_value from grade column...")

            # Try to extract grade value from grade string
            def extract_grade_value(grade_str):
                if pd.isna(grade_str) or not isinstance(grade_str, str):
                    return np.nan
                try:
                    if grade_str.startswith('V'):
                        return float(grade_str.strip()[1:].split('+')[0])
                    return float(grade_str.strip())
                except (ValueError, IndexError):
                    return np.nan

            df['grade_value'] = df['grade'].apply(extract_grade_value)
        elif 'display_difficulty' in df.columns:
            print("Creating grade_value from display_difficulty column...")
            df['grade_value'] = df['display_difficulty'] / 3.0

    # Remove samples with missing grade values
    df = df.dropna(subset=['grade_value'])
    print(f"Using {len(df)} samples with valid grade values")

    # Save original grade labels if available
    grade_labels = df['grade'].values if 'grade' in df.columns else None

    # Remove non-feature columns
    non_feature_cols = ['climb_id', 'name', 'grade', 'setter_username']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Keep grade_value for now (will be removed from features)
    X = df[feature_cols]
    y = df['grade_value'].values

    # Now remove grade_value and display_difficulty from features
    drop_cols = [col for col in ['grade_value', 'display_difficulty'] if col in X.columns]
    X = X.drop(columns=drop_cols)

    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} features")
        print("Filling missing values with zeros")
        X = X.fillna(0)

    # Remove constant columns (these cause issues with correlation and don't add information)
    constant_cols = [col for col in X.columns if X[col].std() == 0]
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns")
        X = X.drop(columns=constant_cols)

    # Get feature names
    feature_names = X.columns.tolist()

    return X, y, feature_names, grade_labels

def split_data(X, y, test_size=0.15, val_size=0.15):
    """
    Split data into training, validation, and test sets.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set

    Returns:
        X_train, X_val, X_test: Feature matrices for each set
        y_train, y_val, y_test: Target variables for each set
    """
    print("Splitting data into training, validation, and test sets...")

    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=np.round(y)
    )

    # Then split the training set into training and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=np.round(y_train_val)
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.

    Args:
        X_train, X_val, X_test: Feature matrices

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled: Scaled feature matrices
        scaler: Fitted StandardScaler
    """
    print("Scaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Neural Network Model
class GradeNN(nn.Module):
    """Neural network model for grade prediction."""

    def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
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

def train_nn_model(X_train, y_train, X_val, y_val, input_size,
                  hidden_sizes=[128, 64, 32], lr=0.001, batch_size=32,
                  epochs=100, patience=10):
    """
    Train a neural network model for grade prediction.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        lr: Learning rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        model: Trained neural network model
        history: Training history
    """
    print("Training neural network model...")

    # Create tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GradeNN(input_size, hidden_sizes)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    best_val_loss = float('inf')
    best_model = None
    counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_mae = mean_absolute_error(y_val, val_outputs.numpy())

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model:
        model.load_state_dict(best_model)

    return model, history

def train_random_forest_model(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest regression model for grade prediction.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        model: Trained Random Forest model
        feature_importances: Feature importance scores
    """
    print("Training Random Forest model...")

    # Parameter grid for grid search - SIMPLIFIED
    param_grid = {
        'n_estimators': [100],  # Just use 100 trees
        'max_depth': [None, 20],  # Try only 2 depth options
        'min_samples_split': [5],  # Fixed value
        'min_samples_leaf': [2]  # Fixed value
    }

    # Initialize model
    base_model = RandomForestRegressor(random_state=42)

    # Perform grid search
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=1
    )

    # Combine training and validation data for grid search
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))

    grid_search.fit(X_train_val, y_train_val)

    # Get best model
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    model = grid_search.best_estimator_
    feature_importances = model.feature_importances_

    return model, feature_importances

def evaluate_model(model, X_test, y_test, is_nn=False, feature_names=None, output_dir=None):
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        is_nn: Whether the model is a neural network
        feature_names: Names of features (for RF model)
        output_dir: Output directory for saving results

    Returns:
        results: Dictionary of evaluation metrics
    """
    print("Evaluating model on test set...")

    # Make predictions
    if is_nn:
        X_test_tensor = torch.FloatTensor(X_test)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy().flatten()
    else:
        y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate accuracy within 1 and 2 grades
    acc_within_1 = np.mean(np.abs(y_pred - y_test) <= 1)
    acc_within_2 = np.mean(np.abs(y_pred - y_test) <= 2)

    # Calculate bias metrics (over/under prediction)
    errors = y_pred - y_test
    over_prediction = np.mean(errors > 0)  # Percentage of times model predicted too high
    under_prediction = np.mean(errors < 0)  # Percentage of times model predicted too low
    exact_prediction = np.mean(errors == 0)  # Percentage of exact predictions

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Accuracy within 1 grade: {acc_within_1:.4f}")
    print(f"Accuracy within 2 grades: {acc_within_2:.4f}")
    print(f"Over-prediction rate: {over_prediction:.4f} ({over_prediction*100:.1f}%)")
    print(f"Under-prediction rate: {under_prediction:.4f} ({under_prediction*100:.1f}%)")
    print(f"Exact prediction rate: {exact_prediction:.4f} ({exact_prediction*100:.1f}%)")

    # Calculate average error by grade
    if output_dir:
        grade_bias_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_pred - y_test
        })
        grade_bias_df['rounded_grade'] = np.round(y_test).astype(int)
        grade_bias = grade_bias_df.groupby('rounded_grade').agg({
            'error': ['mean', 'std', 'count']
        }).reset_index()
        grade_bias.columns = ['grade', 'mean_error', 'std_error', 'count']

        # Only include grades with sufficient samples
        grade_bias = grade_bias[grade_bias['count'] >= 10]

        # Create grade bias plot
        plt.figure(figsize=(10, 6))
        plt.bar(grade_bias['grade'], grade_bias['mean_error'])
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Grade (V-scale)', fontsize=12)
        plt.ylabel('Mean Error (predicted - actual)', fontsize=12)
        plt.title(f'{"Neural Network" if is_nn else "Random Forest"}: Prediction Bias by Grade', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        model_type = "neural_network" if is_nn else "random_forest"
        plt.savefig(os.path.join(output_dir, f"{model_type}_grade_bias.png"), dpi=300)
        plt.close()

        # Save grade bias data
        grade_bias.to_csv(os.path.join(output_dir, f"{model_type}_grade_bias.csv"), index=False)

    # Save metrics to file if output_dir provided
    if output_dir:
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'acc_within_1': acc_within_1,
            'acc_within_2': acc_within_2,
            'over_prediction': over_prediction,
            'under_prediction': under_prediction,
            'exact_prediction': exact_prediction
        }

        model_type = "neural_network" if is_nn else "random_forest"
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.json")

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # Create predictions vs actual plot
    if output_dir:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.title(f'{"Neural Network" if is_nn else "Random Forest"}: Predicted vs Actual Grades', fontsize=14)
        plt.xlabel('Actual Grade Value', fontsize=12)
        plt.ylabel('Predicted Grade Value', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()

        model_type = "neural_network" if is_nn else "random_forest"
        plt.savefig(os.path.join(output_dir, f"{model_type}_predictions.png"), dpi=300)
        plt.close()

        # Save raw predictions for future analysis
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_pred - y_test,
            'abs_error': np.abs(y_pred - y_test),
            'is_over': y_pred > y_test,
            'is_under': y_pred < y_test
        })
        predictions_df.to_csv(os.path.join(output_dir, f"{model_type}_predictions.csv"), index=False)

        # For Random Forest, create feature importance plot
        if not is_nn and feature_names is not None:
            plt.figure(figsize=(12, 10))

            # Sort features by importance
            indices = np.argsort(model.feature_importances_)[::-1]

            # Plot top 20 features (or all if less than 20)
            top_n = min(20, len(feature_names))

            plt.barh(range(top_n), model.feature_importances_[indices[:top_n]])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])

            plt.title('Random Forest Feature Importance', fontsize=14)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
            plt.close()

            # Save feature importance to CSV
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'acc_within_1': acc_within_1,
        'acc_within_2': acc_within_2,
        'over_prediction': over_prediction,
        'under_prediction': under_prediction,
        'exact_prediction': exact_prediction,
        'y_test': y_test,
        'y_pred': y_pred
    }

    return results

def create_simple_report(nn_results, rf_results, feature_names, feature_importances, output_dir):
    """
    Create a simplified report summarizing model training and evaluation.

    Args:
        nn_results: Results from neural network evaluation
        rf_results: Results from random forest evaluation
        feature_names: Names of features
        feature_importances: Feature importance scores from random forest
        output_dir: Output directory for saving report
    """
    print("Creating simple training report...")

    report_path = os.path.join(output_dir, 'training_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Kilter Board Grade Prediction - Model Training Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

        # Model comparison section
        f.write("## Model Comparison\n\n")
        f.write("| Metric | Neural Network | Random Forest |\n")
        f.write("|--------|---------------|---------------|\n")
        f.write(f"| Mean Absolute Error | {nn_results['mae']:.4f} | {rf_results['mae']:.4f} |\n")
        f.write(f"| Root Mean Squared Error | {nn_results['rmse']:.4f} | {rf_results['rmse']:.4f} |\n")
        f.write(f"| R² Score | {nn_results['r2']:.4f} | {rf_results['r2']:.4f} |\n")
        f.write(f"| Accuracy within 1 grade | {nn_results['acc_within_1']*100:.2f}% | {rf_results['acc_within_1']*100:.2f}% |\n")
        f.write(f"| Accuracy within 2 grades | {nn_results['acc_within_2']*100:.2f}% | {rf_results['acc_within_2']*100:.2f}% |\n")
        f.write(f"| Over-prediction rate | {nn_results['over_prediction']*100:.2f}% | {rf_results['over_prediction']*100:.2f}% |\n")
        f.write(f"| Under-prediction rate | {nn_results['under_prediction']*100:.2f}% | {rf_results['under_prediction']*100:.2f}% |\n")
        f.write(f"| Exact prediction rate | {nn_results['exact_prediction']*100:.2f}% | {rf_results['exact_prediction']*100:.2f}% |\n\n")

        # Declare best model
        best_model = "Neural Network" if nn_results['mae'] < rf_results['mae'] else "Random Forest"
        f.write(f"**Best performing model: {best_model}** (based on MAE)\n\n")

        # Model bias
        f.write("## Prediction Bias Analysis\n\n")

        nn_bias = "overestimate" if nn_results['over_prediction'] > nn_results['under_prediction'] else "underestimate"
        rf_bias = "overestimate" if rf_results['over_prediction'] > rf_results['under_prediction'] else "underestimate"

        f.write(f"- The Neural Network tends to {nn_bias} grades ({nn_results['over_prediction']*100:.1f}% over vs {nn_results['under_prediction']*100:.1f}% under).\n")
        f.write(f"- The Random Forest tends to {rf_bias} grades ({rf_results['over_prediction']*100:.1f}% over vs {rf_results['under_prediction']*100:.1f}% under).\n\n")

        # Random Forest section
        f.write("## Random Forest Model\n\n")
        f.write("### Feature Importance\n\n")
        f.write("The Random Forest model identified the following features as most important for grade prediction:\n\n")

        # Sort features by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices[:10]]

        f.write("| Feature | Importance |\n")
        f.write("|---------|------------|\n")
        for feature, importance in top_features:
            f.write(f"| {feature} | {importance:.4f} |\n")

        # Error Analysis
        f.write("\n## Error Analysis\n\n")

        # Neural Network errors
        nn_errors = np.abs(nn_results['y_pred'] - nn_results['y_test'])
        rf_errors = np.abs(rf_results['y_pred'] - rf_results['y_test'])

        f.write("### Distribution of Prediction Errors\n\n")
        f.write("| Error Range | Neural Network | Random Forest |\n")
        f.write("|-------------|---------------|---------------|\n")
        f.write(f"| < 0.5 grade | {np.mean(nn_errors < 0.5)*100:.2f}% | {np.mean(rf_errors < 0.5)*100:.2f}% |\n")
        f.write(f"| < 1.0 grade | {np.mean(nn_errors < 1.0)*100:.2f}% | {np.mean(rf_errors < 1.0)*100:.2f}% |\n")
        f.write(f"| < 1.5 grades | {np.mean(nn_errors < 1.5)*100:.2f}% | {np.mean(rf_errors < 1.5)*100:.2f}% |\n")
        f.write(f"| < 2.0 grades | {np.mean(nn_errors < 2.0)*100:.2f}% | {np.mean(rf_errors < 2.0)*100:.2f}% |\n")
        f.write(f"| >= 2.0 grades | {np.mean(nn_errors >= 2.0)*100:.2f}% | {np.mean(rf_errors >= 2.0)*100:.2f}% |\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        f.write(f"The {best_model} model achieved the best performance with a mean absolute error of {min(nn_results['mae'], rf_results['mae']):.4f} grade points. ")
        f.write(f"Both models were able to predict grades within 1 grade of the actual value for about {min(nn_results['acc_within_1'], rf_results['acc_within_1'])*100:.1f}% of test routes ")
        f.write(f"and within 2 grades for about {min(nn_results['acc_within_2'], rf_results['acc_within_2'])*100:.1f}% of test routes.\n\n")

        # Add bias conclusion
        f.write(f"Both models showed a tendency to {nn_bias if nn_results['over_prediction'] > 0.5 else 'underestimate'} grades, ")
        f.write(f"suggesting that there may be subtle factors affecting difficulty perception that aren't fully captured by the features.\n\n")

        f.write("### Key Findings\n\n")

        # Dynamically generate key findings based on feature importance
        if len(top_features) > 0:
            f.write(f"1. **{top_features[0][0]}** is the most important feature for grade prediction.\n")

        f.write(f"2. The models can predict climbing grades with moderate accuracy (within 1 grade {min(nn_results['acc_within_1'], rf_results['acc_within_1'])*100:.1f}% of the time).\n")

        f.write(f"3. Accuracy increases to {min(nn_results['acc_within_2'], rf_results['acc_within_2'])*100:.1f}% when allowing predictions within 2 grades of the actual value.\n")

        f.write(f"4. Both models tend to {nn_bias if nn_results['over_prediction'] > 0.5 else 'underestimate'} route difficulty, with the {best_model} showing less bias.\n")

        f.write("5. Random Forest provides interpretable results through feature importance analysis.\n")

    print(f"Training report saved to {report_path}")

    # Create error comparison plot
    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 3, 16)  # 0 to 3 grades error in 0.2 increments

    plt.hist(nn_errors, bins=bins, alpha=0.5, label='Neural Network')
    plt.hist(rf_errors, bins=bins, alpha=0.5, label='Random Forest')

    plt.axvline(1.0, color='r', linestyle='--', label='1 Grade Error')
    plt.axvline(2.0, color='g', linestyle='--', label='2 Grade Error')

    plt.xlabel('Absolute Error (in grades)', fontsize=12)
    plt.ylabel('Number of Test Samples', fontsize=12)
    plt.title('Error Distribution Comparison', fontsize=14)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_comparison.png'), dpi=300)
    plt.close()

    # Create bias comparison plot
    plt.figure(figsize=(10, 6))

    nn_errors_raw = nn_results['y_pred'] - nn_results['y_test']
    rf_errors_raw = rf_results['y_pred'] - rf_results['y_test']

    bins = np.linspace(-3, 3, 21)  # -3 to 3 grades error in 0.3 increments

    plt.hist(nn_errors_raw, bins=bins, alpha=0.5, label='Neural Network')
    plt.hist(rf_errors_raw, bins=bins, alpha=0.5, label='Random Forest')

    plt.axvline(0, color='k', linestyle='-', label='No Error')

    plt.xlabel('Error (predicted - actual)', fontsize=12)
    plt.ylabel('Number of Test Samples', fontsize=12)
    plt.title('Prediction Bias Comparison', fontsize=14)
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_comparison.png'), dpi=300)
    plt.close()

def create_prediction_script(output_dir, feature_names):
    """
    Create a script for making predictions with the trained models.

    Args:
        output_dir: Output directory where models are saved
        feature_names: List of feature names used by the models
    """
    print("Creating prediction script...")

    script_path = os.path.join(output_dir, 'predict_grade.py')

    script_content = f"""#!/usr/bin/env python3
\"\"\"
Kilter Board Grade Prediction Script

This script uses the trained models to predict the grade of a Kilter Board route
based on its features.

Usage:
    python predict_grade.py --input climb_features.json --model [nn|rf|ensemble]
\"\"\"

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Feature names used by the models
FEATURE_NAMES = {feature_names}

class GradeNN(nn.Module):
    \"\"\"Neural network model for grade prediction.\"\"\"
    
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
    \"\"\"Load the trained models.\"\"\"
    models = {{}}
    
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
    \"\"\"Convert numerical grade value to string representation.\"\"\"
    grade_int = int(round(grade_value))
    return f"V{{grade_int}}"

def extract_features_from_json(json_file, feature_names):
    \"\"\"Extract features from a JSON file with climb data.\"\"\"
    with open(json_file, 'r') as f:
        climb_data = json.load(f)
    
    # Initialize feature vector with zeros
    features = {{}}
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
        print(f"Warning: Missing features: {{missing_features}}")
        for feature in missing_features:
            df[feature] = 0
    
    return df[feature_names]

def predict_grade(features, models, model_type='ensemble'):
    \"\"\"
    Predict the grade of a climb based on its features.
    
    Args:
        features: DataFrame with climb features
        models: Dictionary of loaded models
        model_type: Type of model to use ('nn', 'rf', or 'ensemble')
        
    Returns:
        Dictionary with prediction results
    \"\"\"
    # Scale features
    X_scaled = models['scaler'].transform(features)
    
    predictions = {{}}
    
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
    
    return {{
        'grade_value': final_prediction,
        'grade': v_grade,
        'model_predictions': predictions
    }}

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
        print("Error: No models found in {{args.model_dir}}")
        return
    
    # Extract features
    features = extract_features_from_json(args.input, FEATURE_NAMES)
    
    # Make prediction
    prediction = predict_grade(features, models, args.model)
    
    # Print results
    print(f"\\nPrediction for climb: {{args.input}}\\n")
    print(f"Predicted grade: {{prediction['grade']}} ({{prediction['grade_value']:.2f}})")
    
    if 'model_predictions' in prediction:
        print("\\nIndividual model predictions:")
        for model, pred_value in prediction['model_predictions'].items():
            print(f"- {{model.upper()}}: {{grade_to_string(pred_value)}} ({{pred_value:.2f}})")

if __name__ == "__main__":
    main()
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Prediction script saved to {script_path}")
    os.chmod(script_path, 0o755)  # Make executable

def main():
    parser = argparse.ArgumentParser(description='Train Kilter Board grade prediction models')
    parser.add_argument('--data_path', type=str, default='dataset/ml_features.csv',
                        help='Path to ML features CSV file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models and results')

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output_dir)

    # Load and preprocess data
    X, y, feature_names, grade_labels = load_and_preprocess_data(args.data_path)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # Train neural network model
    nn_model, nn_history = train_nn_model(
        X_train_scaled, y_train, X_val_scaled, y_val,
        input_size=X_train_scaled.shape[1]
    )

    # Train Random Forest model - simplified for speed
    rf_model, feature_importances = train_random_forest_model(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate models
    nn_results = evaluate_model(nn_model, X_test_scaled, y_test, is_nn=True, output_dir=output_dir)
    rf_results = evaluate_model(rf_model, X_test_scaled, y_test, is_nn=False,
                               feature_names=feature_names, output_dir=output_dir)

    # Create simple training report
    create_simple_report(nn_results, rf_results, feature_names, feature_importances, output_dir)

    # Save models and metadata
    print("Saving models and metadata...")

    # Save neural network model
    torch.save(nn_model.state_dict(), os.path.join(output_dir, 'nn_model.pt'))

    # Save model architecture info
    with open(os.path.join(output_dir, 'nn_architecture.json'), 'w') as f:
        json.dump({
            'input_size': X_train_scaled.shape[1],
            'hidden_sizes': [128, 64, 32]  # Update if you change the architecture
        }, f)

    # Save Random Forest model
    joblib.dump(rf_model, os.path.join(output_dir, 'rf_model.joblib'))

    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    # Create prediction script
    create_prediction_script(output_dir, feature_names)

    print("\nModel training and evaluation complete!")
    print(f"Models and results saved to {output_dir}")
    print("\nKey metrics:")
    print(f"- Neural Network MAE: {nn_results['mae']:.4f}")
    print(f"- Random Forest MAE: {rf_results['mae']:.4f}")
    print(f"- Neural Network accuracy within 1 grade: {nn_results['acc_within_1']*100:.2f}%")
    print(f"- Random Forest accuracy within 1 grade: {rf_results['acc_within_1']*100:.2f}%")
    print(f"- Neural Network accuracy within 2 grades: {nn_results['acc_within_2']*100:.2f}%")
    print(f"- Random Forest accuracy within 2 grades: {rf_results['acc_within_2']*100:.2f}%")
    print(f"- Neural Network prediction bias: {nn_results['over_prediction']*100:.1f}% over, {nn_results['under_prediction']*100:.1f}% under")
    print(f"- Random Forest prediction bias: {rf_results['over_prediction']*100:.1f}% over, {rf_results['under_prediction']*100:.1f}% under")

    best_model = "Neural Network" if nn_results['mae'] < rf_results['mae'] else "Random Forest"
    print(f"\nBest performing model: {best_model}")

    print("\nNext steps:")
    print("1. View the training report for detailed analysis")
    print("2. Use the prediction script to test on new routes")
    print("3. Experiment with model hyperparameters to improve performance")

if __name__ == "__main__":
    main()