import pandas as pd
import joblib

class StockPredictor:
    def __init__(self,model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            print("model loaded successfully")
            return model
        except Exception as e:
            print(f"can't load the model: {e}")
            return None
    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            print("data loaded successfully")
            return df
        except Exception as e:
            print(f"error loading data: {e}")
            return None
    def prepare_input(self, df):
        latest_data = df.iloc[-1:]
        X = latest_data.drop(columns = ['Date', "Close"])
        return X
    def predict(self):
        model = self.load_model()
        df = self.load_data()

        if model is None or df is None:
            return None

        X = self.prepare_input(df)
        print("Input features used")
        print(X)

        prediction = model.predict(X)

        print(f"Predicted next closing price: {prediction[0]}")

        return prediction[0]

if __name__ == "__main__":
    predictor = StockPredictor(
    model_path="models/saved_models/stock_model.pkl",
    data_path="data/processed/featured_data.csv"
    )
    predictor.predict()