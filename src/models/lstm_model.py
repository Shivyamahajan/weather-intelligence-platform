"""
LSTM Deep Learning Model for Rainfall Prediction
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

Architecture:
    Input → LSTM Layer 1 → Dropout → LSTM Layer 2 → 
    Dropout → Dense → Dense → Output (rainfall prediction)

Key concepts:
- SEQUENCE_LENGTH: how many past days the model looks at
- LSTM units: size of hidden state (memory capacity)
- Dropout: randomly turns off neurons to prevent overfitting
- Dense layer: standard fully connected layer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, 
                                      Bidirectional, GRU,
                                      Input, BatchNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CITY            = 'Mumbai'
SEQUENCE_LENGTH = 30      # look at past 30 days to predict next day
BATCH_SIZE      = 64      # process 64 sequences at a time
EPOCHS          = 100     # maximum training epochs
LEARNING_RATE   = 0.001   # how fast the model learns
TEST_SPLIT      = 0.2     # 20% for testing
VAL_SPLIT       = 0.1     # 10% of training for validation

# Weather features to use as input
FEATURES = [
    'precipitation_mm',
    'temp_max_c',
    'temp_min_c',
    'humidity_max_pct',
    'wind_speed_max_kmh',
    'pressure_hpa',
    'cloud_cover_pct',
]


# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────

def load_and_prepare(city=CITY):
    """
    Load data and prepare it for LSTM input.
    
    LSTM requires data in the shape:
    (samples, time_steps, features)
    
    samples = number of training examples
    time_steps = SEQUENCE_LENGTH (30 days)
    features = number of variables (7)
    """
    df = pd.read_csv(
        'data/raw/india_weather_1990_2024.csv',
        parse_dates=['date']
    )
    
    city_df = df[df['city'] == city].sort_values('date').copy()
    
    # Keep only the features we need
    available = [f for f in FEATURES if f in city_df.columns]
    data = city_df[available + ['date']].dropna()
    
    print(f"City: {city}")
    print(f"Records: {len(data):,}")
    print(f"Features: {available}")
    print(f"Sequence length: {SEQUENCE_LENGTH} days")
    
    # Scale features to range [0, 1]
    # LSTM works much better with scaled data
    # MinMaxScaler preserves the relative differences
    feature_data = data[available].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    # Find which column is rainfall (target)
    target_idx = available.index('precipitation_mm')
    
    return scaled_data, scaler, available, target_idx, data['date'].values


def create_sequences(data, seq_length, target_idx):
    """
    Convert time series data into sequences for LSTM.
    
    Example with seq_length=3:
    Original: [1, 2, 3, 4, 5, 6, 7]
    
    Sequences created:
    X[0] = [1, 2, 3]  → y[0] = 4
    X[1] = [2, 3, 4]  → y[1] = 5
    X[2] = [3, 4, 5]  → y[2] = 6
    X[3] = [4, 5, 6]  → y[3] = 7
    
    Each X is a window of past data,
    each y is the next value to predict.
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Input: seq_length days of all features
        X.append(data[i : i + seq_length, :])
        # Target: next day's rainfall only
        y.append(data[i + seq_length, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nSequence shapes:")
    print(f"  X shape: {X.shape}  "
          f"(samples, time_steps, features)")
    print(f"  y shape: {y.shape}  "
          f"(samples,)")
    
    return X, y


def time_based_split(X, y, test_split=TEST_SPLIT, val_split=VAL_SPLIT):
    """Split data chronologically — never random for time series."""
    n = len(X)
    test_size = int(n * test_split)
    val_size  = int(n * val_split)
    train_size = n - test_size - val_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):,} sequences")
    print(f"  Val:   {len(X_val):,} sequences")
    print(f"  Test:  {len(X_test):,} sequences")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────

def build_lstm_model(input_shape, units=64, dropout_rate=0.2):
    """
    Build a stacked LSTM model.
    
    Architecture:
    Input (30 days × 7 features)
        ↓
    LSTM(128 units) — learns temporal patterns
        ↓
    Dropout(0.2) — prevents overfitting
        ↓
    LSTM(64 units) — learns higher-level patterns
        ↓
    Dropout(0.2)
        ↓
    Dense(32) — combines learned features
        ↓
    Dense(1) — final rainfall prediction
    """
    model = Sequential([
        LSTM(128, 
             input_shape=input_shape,
             return_sequences=True,  # pass sequence to next LSTM
             name='lstm_1'),
        Dropout(dropout_rate, name='dropout_1'),
        
        LSTM(64, 
             return_sequences=False,  # last LSTM only returns final output
             name='lstm_2'),
        Dropout(dropout_rate, name='dropout_2'),
        
        Dense(32, activation='relu', name='dense_1'),
        Dense(16, activation='relu', name='dense_2'),
        Dense(1, activation='linear', name='output')
        # linear activation for regression (not sigmoid/softmax)
    ], name='LSTM_Model')
    
    return model


def build_bilstm_model(input_shape, dropout_rate=0.2):
    """
    Build a Bidirectional LSTM model.
    
    Bidirectional LSTM reads the sequence BOTH forward and backward.
    This captures patterns that are more visible in reverse.
    
    Example: In weather, knowing that rainfall INCREASED over
    the past week is captured by forward reading.
    Knowing that tomorrow is PEAK monsoon is better captured
    by patterns leading up to that point.
    """
    model = Sequential([
        Bidirectional(
            LSTM(64, return_sequences=True),
            input_shape=input_shape,
            name='bilstm_1'
        ),
        Dropout(dropout_rate),
        
        Bidirectional(
            LSTM(32, return_sequences=False),
            name='bilstm_2'
        ),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ], name='BiLSTM_Model')
    
    return model


def build_gru_model(input_shape, dropout_rate=0.2):
    """
    Build a GRU (Gated Recurrent Unit) model.
    
    GRU is similar to LSTM but simpler — it has only 2 gates
    instead of LSTM's 3 gates. This makes it:
    - Faster to train
    - Better for smaller datasets
    - Sometimes slightly less accurate but more robust
    """
    model = Sequential([
        GRU(128, 
            input_shape=input_shape,
            return_sequences=True,
            name='gru_1'),
        Dropout(dropout_rate),
        
        GRU(64, 
            return_sequences=False,
            name='gru_2'),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ], name='GRU_Model')
    
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Train a deep learning model with callbacks.
    
    Callbacks explained:
    
    EarlyStopping:
        Stops training if validation loss stops improving.
        This prevents overfitting and saves time.
        patience=15 means stop after 15 epochs of no improvement.
    
    ReduceLROnPlateau:
        Reduces learning rate when training plateaus.
        Learning rate is how big the steps are during learning.
        Smaller steps = more precise but slower learning.
    
    ModelCheckpoint:
        Saves the best version of the model automatically.
        Even if the model gets worse later, the best version is saved.
    """
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='huber',
        # Huber loss is better than MSE for rainfall:
        # - Acts like MSE for small errors (precise)
        # - Acts like MAE for large errors (robust to extreme events)
        metrics=['mae']
    )
    
    model.summary()
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,      # reduce LR by half
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f'models/{model_name.lower().replace(" ","_")}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    print(f"\nTraining {model_name}...")
    print(f"Max epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    print("(EarlyStopping will stop training when validation loss stops improving)")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, scaler, 
                   target_idx, model_name, n_features):
    """
    Evaluate model and convert predictions back to mm.
    
    Important: Our data was scaled to [0,1] before training.
    We must convert predictions BACK to original mm values
    before calculating metrics.
    """
    y_pred_scaled = model.predict(X_test).flatten()
    
    # Inverse transform to get back to original units (mm)
    # We need to reconstruct a full feature array to use inverse_transform
    dummy = np.zeros((len(y_pred_scaled), n_features))
    dummy[:, target_idx] = y_pred_scaled
    y_pred_mm = scaler.inverse_transform(dummy)[:, target_idx]
    
    dummy2 = np.zeros((len(y_test), n_features))
    dummy2[:, target_idx] = y_test
    y_true_mm = scaler.inverse_transform(dummy2)[:, target_idx]
    
    # Clip negative predictions
    y_pred_mm = np.clip(y_pred_mm, 0, None)
    
    mae  = mean_absolute_error(y_true_mm, y_pred_mm)
    rmse = np.sqrt(mean_squared_error(y_true_mm, y_pred_mm))
    r2   = r2_score(y_true_mm, y_pred_mm)
    
    print(f"\n{model_name} Test Results:")
    print(f"  MAE:  {mae:.4f} mm")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'Model': model_name,
        'MAE':  round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2':   round(r2, 4)
    }, y_pred_mm, y_true_mm


def plot_training_history(histories, model_names):
    """Plot training and validation loss for all models."""
    fig, axes = plt.subplots(1, len(histories), 
                              figsize=(7 * len(histories), 5))
    if len(histories) == 1:
        axes = [axes]
    
    for ax, history, name in zip(axes, histories, model_names):
        ax.plot(history.history['loss'], 
                label='Training Loss', color='steelblue', linewidth=2)
        ax.plot(history.history['val_loss'], 
                label='Validation Loss', color='red', linewidth=2)
        
        best_epoch = np.argmin(history.history['val_loss'])
        ax.axvline(x=best_epoch, color='green', 
                   linestyle='--', alpha=0.7,
                   label=f'Best Epoch: {best_epoch+1}')
        
        ax.set_title(f'{name}\nTraining History', 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/dl_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history plots saved!")


def plot_dl_predictions(predictions_dict, y_true, model_names):
    """Plot actual vs predicted for all deep learning models."""
    n_show = min(365, len(y_true))
    
    fig, axes = plt.subplots(len(model_names), 1, 
                              figsize=(16, 5 * len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    
    colors = ['red', 'darkorange', 'purple']
    
    for ax, name, color in zip(axes, model_names, colors):
        y_pred = predictions_dict[name]
        
        ax.plot(range(n_show), y_true[-n_show:],
                label='Actual', color='steelblue',
                linewidth=1.0, alpha=0.8)
        ax.plot(range(n_show), y_pred[-n_show:],
                label=f'Predicted ({name})', color=color,
                linewidth=1.0, alpha=0.7, linestyle='--')
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        
        ax.set_title(f'{name} — Actual vs Predicted\n'
                     f'RMSE={rmse:.3f}mm | R²={r2:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Days (last 365 days shown)')
        ax.set_ylabel('Rainfall (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/dl_predictions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("DL prediction plots saved!")


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DEEP LEARNING — LSTM / BiLSTM / GRU")
    print(f"City: {CITY} | Sequence Length: {SEQUENCE_LENGTH} days")
    print("=" * 60)
    
    # Step 1: Prepare data
    scaled_data, scaler, features, target_idx, dates = load_and_prepare()
    n_features = len(features)
    
    # Step 2: Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH, target_idx)
    
    # Step 3: Split
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)
    
    # Input shape for all models
    input_shape = (SEQUENCE_LENGTH, n_features)
    print(f"\nInput shape: {input_shape}")
    
    # Step 4: Build models
    models_to_train = {
        'LSTM':   build_lstm_model(input_shape),
        'BiLSTM': build_bilstm_model(input_shape),
        'GRU':    build_gru_model(input_shape),
    }
    
    # Step 5: Train and evaluate all models
    all_results   = []
    all_histories = []
    predictions   = {}
    
    for model_name, model in models_to_train.items():
        print(f"\n{'='*55}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*55}")
        
        history = train_model(
            model, X_train, y_train, X_val, y_val, model_name
        )
        all_histories.append(history)
        
        result, y_pred_mm, y_true_mm = evaluate_model(
            model, X_test, y_test, scaler,
            target_idx, model_name, n_features
        )
        
        all_results.append(result)
        predictions[model_name] = y_pred_mm
        
        # Save model
        model.save(f'models/{model_name.lower()}_model.keras')
        print(f"Model saved: models/{model_name.lower()}_model.keras")
    
    # Step 6: Visualize
    plot_training_history(all_histories, list(models_to_train.keys()))
    plot_dl_predictions(predictions, y_true_mm,
                        list(models_to_train.keys()))
    
    # Step 7: Final comparison
    print("\n" + "="*55)
    print("DEEP LEARNING MODELS COMPARISON")
    print("="*55)
    results_df = pd.DataFrame(all_results).sort_values('RMSE')
    print(results_df.to_string(index=False))
    
    results_df.to_csv('reports/dl_model_results.csv', index=False)
    print("\n✅ Deep learning training complete!")
    print("Check reports/figures/ for training history and prediction plots.")