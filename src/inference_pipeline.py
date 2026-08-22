import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path

sys.path.append("..")
from src.logger import Logger

ROOT_DIR = Path(__file__).resolve().parent.parent


class InferencePipeline:
    def __init__(self, log_file: str = "inference.log"):
        self.logger = Logger(log_file)

        artifacts_dir = ROOT_DIR / "models" / "artifacts"
        model_path = ROOT_DIR / "models" / "improved" / "et_improved_model.pkl"

        self.model = joblib.load(model_path)
        self.encoders = joblib.load(artifacts_dir / "encoders.pkl")
        self.scalers = joblib.load(artifacts_dir / "scalers.pkl")
        self.train_columns = joblib.load(artifacts_dir / "train_columns.pkl")
        self.threshold = joblib.load(artifacts_dir / "threshold.pkl")

        self.logger.info("InferencePipeline initialized: model, encoders, scalers, columns, threshold loaded")

    def _create_features(self, df):
        df = df.copy()

        season_dict = {
            'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
            'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
            'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
            'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
        }
        df['fe_season'] = df['arrival_date_month'].map(season_dict).fillna('Unknown')

        df["fe_total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        df["fe_is_weekend_stay"] = (df["stays_in_weekend_nights"] > 0).astype(int)
        df["fe_weekend_ratio"] = df["stays_in_weekend_nights"] / (df["fe_total_nights"] + 1)
        df["fe_total_guests"] = df["adults"] + df["children"] + df["babies"]
        df["fe_room_type_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
        df["fe_has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
        df["fe_has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
        df["fe_has_deposit"] = (df["deposit_type"] != "No Deposit").astype(int)
        df["fe_has_parking_request"] = (df["required_car_parking_spaces"] > 0).astype(int)
        df["fe_has_special_request"] = (df["total_of_special_requests"] > 0).astype(int)

        return df

    def _encode(self, df):
        df = df.copy()
        for col, le in self.encoders.items():
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        df = df.reindex(columns=self.train_columns, fill_value=0)
        return df

    def _scale(self, df):
        df = df.copy()
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        return df

    def predict_one(self, raw_input: dict):
        """
        Predicts a single booking from a dict of raw input values.
        """
        self.logger.info(f"Received raw input: {raw_input}")

        df = pd.DataFrame([raw_input])

        if df.loc[0, "meal"] == "Undefined":
            df.loc[0, "meal"] = "no_meal_type"

        df = self._create_features(df)
        df = self._encode(df)
        df = self._scale(df)

        proba = self.model.predict_proba(df)[:, 1][0]
        prediction = int(proba >= self.threshold)

        self.logger.info(f"Prediction: {prediction}, Probability: {proba:.4f}, Threshold: {self.threshold}")

        return {
            "prediction": prediction,
            "probability": float(proba),
            "threshold": float(self.threshold)
        }

    def predict_batch(self, df: pd.DataFrame):
        """
        Predicts for a batch of bookings given as a DataFrame with raw columns.
        """
        self.logger.info(f"Received batch input with shape: {df.shape}")

        df = df.copy()
        df.loc[df["meal"] == "Undefined", "meal"] = "no_meal_type"

        df = self._create_features(df)
        df_encoded = self._encode(df)
        df_scaled = self._scale(df_encoded)

        proba = self.model.predict_proba(df_scaled)[:, 1]
        predictions = (proba >= self.threshold).astype(int)

        self.logger.info(f"Generated {len(predictions)} predictions")

        result_df = df.copy()
        result_df["cancellation_probability"] = proba
        result_df["prediction"] = predictions

        return result_df