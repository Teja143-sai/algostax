import pandas as pd
import pandas_ta as ta
import os


def add_technical_indicators(input_filename, output_filename):
    print(f"Loading raw data from {input_filename}...")

    # Load the data
    df = pd.read_csv(input_filename, index_col=0, parse_dates=True)

    # Clean up multi-level columns if yfinance added them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- THE FIX: FORCE COLUMNS TO BE NUMBERS ---
    # This loops through every column and forces it to be a number.
    # If it finds text (like "^NSEI" in a hidden row), it turns it into a blank (NaN).
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Delete any rows that became blank
    df.dropna(inplace=True)

    print("Calculating technical indicators...")

    # Trend: Exponential Moving Averages (EMA)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)

    # Momentum: Relative Strength Index (RSI)
    df.ta.rsi(length=14, append=True)

    # Momentum: MACD
    df.ta.macd(append=True)

    # Volatility: Average True Range (ATR)
    df.ta.atr(length=14, append=True)

    # Clean up lagging empty rows caused by the EMA/MACD math
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"Cleaned up {initial_rows - final_rows} empty rows created by indicator lag.")

    # Save the new supercharged dataset
    df.to_csv(output_filename)
    print(f"Success! Engineered data saved to {output_filename}")


# --- CONFIGURATION (Smart Absolute Paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "historical_market_data.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "engineered_market_data.csv")

# Run the function
add_technical_indicators(INPUT_FILE, OUTPUT_FILE)