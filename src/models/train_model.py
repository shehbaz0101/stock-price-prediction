import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

class StockModelTrainer:
    def __init__(self, input_path : str, model_path : str):
        self.input_path = input_path
        self.model_path = model_path
        
    def load_data(self):
        try:
            df = pd.read_csv(self.input_path)
            print("data loaded successfully")
            return df
        except Exception as e:
            print(f"data not found : {e}")
            return None
        
    def split_data(self, df):
        split_index = int(len(df) * .8)
        
        train = df[:split_index]
        test = df[split_index:]
        
        return train, test
    
    def prepare_features(self,df):
        X = df.drop(columns = ['Date', 'Close'])
        y = df['Close']
        
        return X,y
    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train,y_train)
        print("Model trained successfully")
        return model
    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions) **.5
        
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
    
    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
        joblib.dump(model, self.model_path)
        print(f"model saved at: {self.model_path}")
        
    def run_pipeline(self):
        df = self.load_data()
        
        if df is not None:
            train, test = self.split_data(df)
            print("train size:", len(train))
            print("test size:", len(test))
            
            X_train, y_train = self.prepare_features(train)
            X_test, y_test = self.prepare_features(test)
            
            model = self.train_model(X_train, y_train)
            self.evaluate_model = self.evaluate_model(model,X_test, y_test)
            self.save_model(model)
            
            
if __name__ == "__main__":
    trainer = StockModelTrainer(
    input_path = "data/processed/featured_data.csv",
    model_path = "models/saved_models/stock_model.pkl")
    trainer.run_pipeline()