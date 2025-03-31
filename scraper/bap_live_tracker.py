import yfinance as yf
import pandas as pd
import logging
import os
import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, List


# Configure logging
def setup_logging() -> None:
    """Set up logging configuration with file and console handlers."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "stock_data.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


# Decorators
def log_execution_time(func):
    """Decorator to measure and log the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info(f"Starting execution of {func.__name__}")
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")

        return result

    return wrapper


def handle_yf_errors(func):
    """Decorator to catch and handle yfinance-related errors."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(__name__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.warning("Returning empty DataFrame due to error")
            return pd.DataFrame()

    return wrapper


# Main functionality
@handle_yf_errors
@log_execution_time
def fetch_stock_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stock data from Yahoo Finance for the given ticker.

    Args:
        ticker: The stock ticker symbol

    Returns:
        Dictionary containing stock information or None if data retrieval fails
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Fetching data for ticker: {ticker}")

    stock = yf.Ticker(ticker)
    data = stock.info

    if not data:
        logger.warning(f"No data retrieved for ticker: {ticker}")
        return None

    logger.info(f"Successfully retrieved data for ticker: {ticker}")
    return data


def convert_to_dataframe(data: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert stock data dictionary to a pandas DataFrame.

    Args:
        data: Dictionary containing stock information

    Returns:
        DataFrame containing the stock data
    """
    logger = logging.getLogger(__name__)

    if data is None:
        logger.warning("No data to convert to DataFrame")
        return pd.DataFrame()

    try:
        df = pd.DataFrame([data])
        logger.info("Successfully converted data to DataFrame")
        return df
    except Exception as e:
        logger.error(f"Error converting data to DataFrame: {str(e)}")
        return pd.DataFrame()


def save_to_csv(df: pd.DataFrame, ticker: str) -> bool:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        ticker: Ticker symbol for filename generation

    Returns:
        Boolean indicating success or failure
    """
    logger = logging.getLogger(__name__)

    if df.empty:
        logger.warning(f"Empty DataFrame for {ticker}, nothing to save")
        return False

    # Replace dots in ticker with underscores for filename
    safe_ticker = ticker.replace('.', '_')
    csv_filename = f"{safe_ticker}_stockdata.csv"

    try:
        # Create data directory if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file_path = os.path.join(data_dir, csv_filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Data successfully saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to CSV for {ticker}: {str(e)}")
        return False


def process_ticker(ticker: str) -> bool:
    """
    Process a single ticker: fetch data, convert to DataFrame, and save to CSV.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Boolean indicating success or failure
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing ticker: {ticker}")

    # Fetch data
    data = fetch_stock_data(ticker)

    # Convert to DataFrame
    df = convert_to_dataframe(data)

    # Save to CSV
    if not df.empty:
        success = save_to_csv(df, ticker)
        if success:
            logger.info(f"All data for {ticker} has been successfully processed and saved")
            return True
        else:
            logger.error(f"Failed to save data for {ticker}")
            return False
    else:
        logger.warning(f"No data to save for {ticker}")
        return False


@log_execution_time
def process_multiple_tickers(tickers: List[str]) -> Dict[str, bool]:
    """
    Process multiple ticker symbols, fetching and saving data for each.

    Args:
        tickers: List of stock ticker symbols

    Returns:
        Dictionary mapping tickers to their processing success status
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing for {len(tickers)} tickers: {', '.join(tickers)}")

    results = {}

    for ticker in tickers:
        success = process_ticker(ticker)
        results[ticker] = success

    # Log summary
    successful = sum(1 for status in results.values() if status)
    logger.info(f"Batch processing complete. Successfully processed {successful}/{len(tickers)} tickers")

    return results


def main(tickers: List[str]) -> None:
    """
    Main function to orchestrate the stock data retrieval and saving process for multiple tickers.

    Args:
        tickers: List of stock ticker symbols
    """
    # Set up logging
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info(f"Starting stock data retrieval process for {len(tickers)} tickers")

    # Process all tickers
    results = process_multiple_tickers(tickers)

    # Print summary to console
    print("\nProcessing Summary:")
    print("------------------")
    for ticker, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{ticker}: {status}")


if __name__ == "__main__":
    # Define tickers
    ticker_symbols = ["BAP", "ILF", "BRK-B"]  # Note: Yahoo Finance uses hyphen for BRK.B

    # Run main process
    main(ticker_symbols)

