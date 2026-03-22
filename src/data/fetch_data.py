import os
import yfinance as yf
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self, ticker: str, period_years: int = 5, save_dir: str = "data/raw"):
        self.ticker = ticker
        self.period_years = period_years
        self.save_dir = save_dir
    # Generate file name dynamically
        self.file_path = os.path.join(self.save_dir, f"{self.ticker}_data.csv")

    def _get_date_range(self):
    #Private method: calculates start and end dates
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * self.period_years)
        return start_date, end_date

    def fetch_data(self):
    #Fetch stock data from yfinance
        try:
            start_date, end_date = self._get_date_range()

            print(f"Fetching data for {self.ticker}...")
            print(f"From {start_date.date()} to {end_date.date()}")

            df = yf.download(self.ticker, start=start_date, end=end_date)

            if df.empty:
                raise ValueError("No data fetched. Check ticker symbol.")

            df.reset_index(inplace=True)

            return df

        except Exception as e:
            print(f"Error while fetching data: {e}")
            return None

    def save_data(self, df):
    #Save dataframe to CSV
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            df.to_csv(self.file_path, index=False)
            print(f"Data saved at: {self.file_path}")
        except Exception as e:
            print(f"Error while saving data: {e}")

    def run_pipeline(self):
    #Complete pipeline: fetch + save
    
        df = self.fetch_data()
        if df is not None:
            self.save_data(df)
            print(df.head())

# Entry point

if __name__ == "__main__":
    fetcher = StockDataFetcher(ticker="RELIANCE.NS")
    fetcher.run_pipeline()
