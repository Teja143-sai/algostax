import yfinance as yf
import pandas as pd


def download_stock_data(ticker_symbol, start_date, end_date, output_filename):
    """
    Downloads historical daily OHLCV data for a given ticker and saves it to a CSV file.
    """
    print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}...")

    # Fetch data using yfinance
    # For Indian markets like Nifty 50, use '^NSEI'. For Gold futures, use 'GC=F'.
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if data.empty:
        print("No data found. Please check the ticker symbol or date range.")
        return

    # Save the data to a CSV file
    data.to_csv(output_filename)
    print(f"Success! Data saved to {output_filename}")
    print(f"Total trading days collected: {len(data)}")


# --- CONFIGURATION ---
# Change '^NSEI' to 'GC=F' if gold data is preferred for the initial test.
TICKER = "^NSEI"
START = "2015-01-01"
END = "2026-01-01"
OUTPUT_FILE = "historical_market_data.csv"

# Run the data download function
download_stock_data(TICKER, START, END, OUTPUT_FILE)