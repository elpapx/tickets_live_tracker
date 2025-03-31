import yfinance as yf
import pandas as pd
import logging
import os
import time
import functools
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import json
import sys

from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime, Text, Integer, insert


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


# Database functions
def setup_database(db_uri: str) -> Tuple[Any, Any]:
    """
    Set up the database connection.

    Args:
        db_uri: Database connection URI

    Returns:
        Tuple of (engine, metadata)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up database connection to {db_uri}")

    try:
        engine = create_engine(db_uri)
        metadata = MetaData()

        # Reflect existing tables
        metadata.reflect(bind=engine)

        logger.info("Database connection established successfully")
        return engine, metadata
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        raise


def import_stock_data_to_db(data: Dict[str, Any], ticker: str, engine, metadata) -> bool:
    """
    Import stock data directly to the database.

    Args:
        data: Dictionary of stock data
        ticker: Ticker symbol
        engine: SQLAlchemy engine
        metadata: SQLAlchemy metadata

    Returns:
        Boolean indicating success or failure
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Importing data for ticker {ticker} to database")

    try:
        # Get the finance_data table
        if 'finance_data' not in metadata.tables:
            logger.error("finance_data table not found in database")
            return False

        finance_data_table = metadata.tables['finance_data']

        # Filter and prepare data for database insertion
        timestamp = datetime.now()

        # Calculate common financial metrics if not present
        if 'marketCap' in data and 'sharesOutstanding' in data and data['sharesOutstanding'] > 0:
            if 'bookValue' not in data:
                data['bookValue'] = data.get('totalAssets', 0) - data.get('totalLiabilities', 0)
                data['bookValue'] = data['bookValue'] / data['sharesOutstanding'] if data[
                                                                                         'sharesOutstanding'] > 0 else 0

            if 'priceToBook' not in data and data.get('bookValue', 0) > 0:
                data['priceToBook'] = data.get('currentPrice', 0) / data['bookValue']

        # Create a new record
        stock_data = {
            'ticker': ticker,
            'date_updated': timestamp,
            'price': data.get('currentPrice', data.get('regularMarketPrice', 0)),
            'previous_close': data.get('previousClose', 0),
            'open_price': data.get('open', 0),
            'day_high': data.get('dayHigh', 0),
            'day_low': data.get('dayLow', 0),
            'volume': data.get('volume', 0),
            'market_cap': data.get('marketCap', 0),
            'beta': data.get('beta', 0),
            'pe_ratio': data.get('trailingPE', 0),
            'eps': data.get('trailingEps', 0),
            'dividend_yield': data.get('dividendYield', 0) * 100 if data.get('dividendYield') else 0,
            'fifty_two_week_high': data.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': data.get('fiftyTwoWeekLow', 0),
            'sector': data.get('sector', ''),
            'industry': data.get('industry', ''),
            'company_name': data.get('longName', data.get('shortName', ticker)),
            'book_value': data.get('bookValue', 0),
            'price_to_book': data.get('priceToBook', 0),
            'raw_data': json.dumps(data)  # Store full data as JSON
        }

        # Insert data
        with engine.connect() as conn:
            stmt = insert(finance_data_table).values(**stock_data)
            conn.execute(stmt)
            conn.commit()

        logger.info(f"Successfully inserted data for {ticker} into finance_data table")

        # Update or insert into bvl_stocks table if it exists
        if 'bvl_stocks' in metadata.tables:
            bvl_stocks_table = metadata.tables['bvl_stocks']

            # Check if ticker already exists
            with engine.connect() as conn:
                result = conn.execute(f"SELECT id FROM bvl_stocks WHERE ticker = '{ticker}'").fetchone()

                if result:
                    # Update existing record
                    stmt = f"""
                    UPDATE bvl_stocks 
                    SET price = {stock_data['price']},
                        previous_close = {stock_data['previous_close']},
                        last_updated = '{timestamp}',
                        company_name = '{stock_data['company_name']}'
                    WHERE ticker = '{ticker}'
                    """
                    conn.execute(stmt)
                else:
                    # Insert new record
                    stmt = f"""
                    INSERT INTO bvl_stocks (ticker, company_name, price, previous_close, last_updated)
                    VALUES ('{ticker}', '{stock_data['company_name']}', {stock_data['price']}, 
                            {stock_data['previous_close']}, '{timestamp}')
                    """
                    conn.execute(stmt)

                conn.commit()

            logger.info(f"Successfully updated bvl_stocks table for {ticker}")

        return True

    except Exception as e:
        logger.error(f"Error importing data to database: {str(e)}")
        return False


def import_csv_to_db(csv_path: str, ticker: str, engine, metadata) -> bool:
    """
    Import data from CSV file to database.

    Args:
        csv_path: Path to CSV file
        ticker: Ticker symbol
        engine: SQLAlchemy engine
        metadata: SQLAlchemy metadata

    Returns:
        Boolean indicating success or failure
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Importing CSV data from {csv_path} to database for ticker {ticker}")

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        if df.empty:
            logger.warning(f"CSV file {csv_path} is empty")
            return False

        # Convert DataFrame row to dictionary
        data_dict = df.iloc[0].to_dict()

        # Import to database
        return import_stock_data_to_db(data_dict, ticker, engine, metadata)

    except Exception as e:
        logger.error(f"Error importing CSV to database: {str(e)}")
        return False


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


def save_to_csv(df: pd.DataFrame, ticker: str) -> Optional[str]:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        ticker: Ticker symbol for filename generation

    Returns:
        Path to saved CSV file or None if saving failed
    """
    logger = logging.getLogger(__name__)

    if df.empty:
        logger.warning(f"Empty DataFrame for {ticker}, nothing to save")
        return None

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
        return file_path
    except Exception as e:
        logger.error(f"Error saving data to CSV for {ticker}: {str(e)}")
        return None


def process_ticker(ticker: str, engine=None, metadata=None) -> Dict[str, Any]:
    """
    Process a single ticker: fetch data, convert to DataFrame, save to CSV, and optionally to database.

    Args:
        ticker: Stock ticker symbol
        engine: SQLAlchemy engine (optional)
        metadata: SQLAlchemy metadata (optional)

    Returns:
        Dictionary with results of the processing
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing ticker: {ticker}")

    result = {
        "ticker": ticker,
        "csv_success": False,
        "db_success": False,
        "csv_path": None
    }

    # Fetch data
    data = fetch_stock_data(ticker)

    # Check if data was successfully retrieved
    if not data:
        return result

    # Convert to DataFrame
    df = convert_to_dataframe(data)

    # Save to CSV
    if not df.empty:
        csv_path = save_to_csv(df, ticker)
        if csv_path:
            result["csv_success"] = True
            result["csv_path"] = csv_path
            logger.info(f"CSV file created successfully for {ticker}")

            # Import to database if engine and metadata are provided
            if engine and metadata:
                db_success = import_stock_data_to_db(data, ticker, engine, metadata)
                result["db_success"] = db_success
                if db_success:
                    logger.info(f"Data for {ticker} successfully imported to database")
                else:
                    logger.error(f"Failed to import data for {ticker} to database")
        else:
            logger.error(f"Failed to save CSV for {ticker}")
    else:
        logger.warning(f"No data to save for {ticker}")

    return result


@log_execution_time
def process_multiple_tickers(tickers: List[str], db_uri: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process multiple ticker symbols, fetching and saving data for each.

    Args:
        tickers: List of stock ticker symbols
        db_uri: Database connection URI (optional)

    Returns:
        List of result dictionaries for each ticker
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting batch processing for {len(tickers)} tickers: {', '.join(tickers)}")

    engine = None
    metadata = None

    # Set up database connection if URI is provided
    if db_uri:
        try:
            engine, metadata = setup_database(db_uri)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            logger.warning("Continuing without database integration")

    results = []

    for ticker in tickers:
        result = process_ticker(ticker, engine, metadata)
        results.append(result)

    # Log summary
    successful_csv = sum(1 for r in results if r["csv_success"])
    successful_db = sum(1 for r in results if r["db_success"])

    logger.info(f"Batch processing complete:")
    logger.info(f"  - CSV files created: {successful_csv}/{len(tickers)}")
    logger.info(f"  - Database imports: {successful_db}/{len(tickers)}")

    return results


def main():
    """Main function that handles PostgreSQL connection."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get tickers from command line or use defaults
    if len(sys.argv) > 1:
        ticker_symbols = [t.strip() for t in sys.argv[1].split(',')]
    else:
        ticker_symbols = ["BAP", "ILF", "BRK-B"]  # Default tickers

    logger.info(f"Starting stock data retrieval for tickers: {', '.join(ticker_symbols)}")

    # PostgreSQL connection settings
    db_uri = "postgresql://postgres@localhost:5432/tickets_live_tracker"

    # Process tickers
    results = process_multiple_tickers(ticker_symbols, db_uri)

    # Print summary to console
    print("\nProcessing Summary:")
    print("------------------")
    for result in results:
        ticker = result["ticker"]
        csv_status = "Success" if result["csv_success"] else "Failed"
        db_status = "Success" if result["db_success"] else "Failed"

        print(f"{ticker}:")
        print(f"  - CSV: {csv_status}")
        print(f"  - Database: {db_status}")
        if result["csv_path"]:
            print(f"  - CSV Path: {result['csv_path']}")
        print()


if __name__ == "__main__":
    main()