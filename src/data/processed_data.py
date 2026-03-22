import pandas as pd

class DataPreprocessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        """Load raw CSV data"""
        try:
            df = pd.read_csv(self.input_path)
            print("Data loaded successfully")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_data(self, df):
        """Perform basic cleaning"""
        try:
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Sort by date
            df = df.sort_values(by='Date')
            
            #creating columns with exact date and year
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day

            # Drop missing values
            df = df.dropna()

            print("Data cleaned successfully")
            return df
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return None

    def save_data(self, df):
        """Save processed data"""
        try:
            df.to_csv(self.output_path, index=False)
            print(f"Processed data saved at: {self.output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def run_pipeline(self):
        """Full preprocessing pipeline"""
        df = self.load_data()
        if df is not None:
            df = self.clean_data(df)
            if df is not None:
                self.save_data(df)
                print(df.head())

if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        input_path="data/raw/RELIANCE.NS_data.csv",
        output_path="data/processed/cleaned_data.csv"
    )
    preprocessor.run_pipeline()
