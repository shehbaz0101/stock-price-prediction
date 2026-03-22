import pandas as pd

class FeatureEngineer:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        try:
            df = pd.read_csv(self.input_path)
            print("Data loaded successfully")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def create_features(self, df):
        try:
            # Convert Date
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')

            # 🔥 Lag Features
            df['lag_1'] = df['Close'].shift(1)
            df['lag_2'] = df['Close'].shift(2)
            df['lag_3'] = df['Close'].shift(3)
            
            df['price_change'] = df['Close'] - df['lag_1']

            # 🔥 Moving Averages
            df['ma_7'] = df['Close'].rolling(window=7).mean()
            df['ma_14'] = df['Close'].rolling(window=14).mean()

            # 🔥 Rolling Standard Deviation (Volatility)
            df['std_7'] = df['Close'].rolling(window=7).std()

            # Drop NA (created due to shifting/rolling)
            df = df.dropna()

            print("Features created successfully")
            return df

        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return None

    def save_data(self, df):
        try:
            df.to_csv(self.output_path, index=False)
            print(f"Feature data saved at: {self.output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def run_pipeline(self):
        df = self.load_data()
        if df is not None:
            df = self.create_features(df)
            if df is not None:
                self.save_data(df)
                print(df.head())

if __name__ == "__main__":
    fe = FeatureEngineer(
    input_path="data/processed/cleaned_data.csv",
    output_path="data/processed/featured_data.csv"
    )
    fe.run_pipeline()
