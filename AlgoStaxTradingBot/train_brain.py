import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_trading_ai(input_filename):
    print(f"Loading engineered dataset from {input_filename}...")

    # FIX: Changed index_col="Date" to index_col=0
    df = pd.read_csv(input_filename, index_col=0, parse_dates=True)

    # 1. Create the Target Variable (What the AI tries to predict)
    # If tomorrow's close is higher than today's close, Target = 1 (UP). Otherwise 0 (DOWN).
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Drop the very last row because we don't know the future day's close yet!
    df.dropna(inplace=True)

    # 2. Select Features for the AI to learn from
    feature_cols = ['EMA_20', 'EMA_50', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'ATRr_14']
    X = df[feature_cols]
    y = df['Target']

    # 3. Train/Test Split (Chronological Order)
    # We take the first 80% of history for training, and reserve the last 20% for testing.
    split_index = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

    # 4. Initialize and Train the Random Forest
    print("Training the Random Forest model (this may take a few seconds)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # 5. Evaluate the model on unseen test data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n================ AI PERFORMANCE EVALUATION ================")
    print(f"Model Directional Accuracy: {accuracy * 100:.2f}%")
    print("===========================================================")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 6. Test a mock real-time prediction using the last row of data
    last_day_features = X.iloc[[-1]]
    probabilities = model.predict_proba(last_day_features)[0]

    print("\n🔮 LIVE TEST PREDICTION FOR NEXT TRADING DAY:")
    print(f"🔴 DOWN Probability: {probabilities[0] * 100:.2f}%")
    print(f"🟢 UP Probability: {probabilities[1] * 100:.2f}%")


# --- CONFIGURATION (Smart Absolute Paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "engineered_market_data.csv")

# Run the training function
train_trading_ai(INPUT_FILE)