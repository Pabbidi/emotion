import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
import os

# Safe logging directory creation
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename='logs/ingestion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_log_dir(log_dir: str = "logs"):
    """Ensure log directory exists."""
    os.makedirs(log_dir, exist_ok=True)

def load_params(file_path: str) -> dict:
    """Load parameters from YAML config."""
    try:
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully from %s", file_path)
        return params
    except Exception as e:
        logging.error("Failed to load parameters: %s", e)
        raise

def load_dataset(url: str) -> pd.DataFrame:
    """Load dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully from %s", url)
        return df
    except Exception as e:
        logging.error("Failed to load dataset: %s", e)
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataset: drop tweet_id and encode sentiment."""
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        filtered_df['sentiment'] = filtered_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Preprocessing completed. Rows after filtering: %d", len(filtered_df))
        return filtered_df
    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        raise

def split_and_save_dataset(df: pd.DataFrame, test_size: float, train_path: str, test_path: str):
    """Split data and save to CSV files."""
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info("Data split and saved to %s and %s", train_path, test_path)
    except Exception as e:
        logging.error("Failed to split and save dataset: %s", e)
        raise

if __name__ == "__main__":
    create_log_dir()
    try:
        params = load_params("params.yaml")
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(url)
        processed_df = preprocess_dataset(df)
        split_and_save_dataset(
            processed_df,
            test_size=params["data_ingestion"]["test_size"],
            train_path="data/raw/train.csv",
            test_path="data/raw/test.csv"
        )
    except Exception as e:
        logging.critical("Pipeline failed: %s", e)
