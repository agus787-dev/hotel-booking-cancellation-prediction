import pandas as pd
import numpy as np
import sys
import joblib

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.logger import Logger


sys.path.append("..")
ROOT_DIR = Path(__file__).resolve().parent.parent


class FeatureEngineer:
    def __init__(self, log_file: str):
        self.logger = Logger(log_file)
        self.scaler = StandardScaler()
        self.encoders = {}
        self.logger.info("Preprocessor initialized")
        
        self.output_dir_path = ROOT_DIR / "data" / "preprocessed"
        self.output_dir_path.mkdir(parents=True, exist_ok=True)


    def drop_and_handling_missing_values(self, df):
        self.logger.info("Starting column dropping process...")

        initial_rows = df.shape[0]

        # Company => too many missing values (94.3%) and ID
        # Reservation status => This column is related to the target value
        # Agent => Id => It does not have predictive power
        # reservation_status_date => it is not time format
        # Arrival date year => 1 unique value (2024)
        drop_cols = ["company", "reservation_status", "agent", "reservation_status_date", "arrival_date_year"]
        self.logger.info(f"Dropping columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)
        self.logger.info(f"Remaining columns after drop: {df.shape[1]}")

        # meal
        undefined_meal_count = (df['meal'] == 'Undefined').sum()
        df['meal'] = df['meal'].replace('Undefined', 'no_meal_type')
        self.logger.info(f"Replaced 'Undefined' with 'no_meal_type' in 'meal' column ({undefined_meal_count} rows affected)")

        # 4 missing values
        before_rows = df.shape[0]
        df = df.dropna(subset=["children"])
        dropped = before_rows - df.shape[0]
        self.logger.info(f"Dropped {dropped} rows with missing 'children' values")

        # adults
        before_rows = df.shape[0]
        df = df[df["adults"] <= 5]
        dropped = before_rows - df.shape[0]
        self.logger.info(f"Dropped {dropped} rows where 'adults' > 5 (outliers)")

        # If `adults = 0` while `children` or `babies` is greater than 0, this should be considered
        # an invalid or illogical case, because children or babies cannot normally make a hotel
        # reservation without an adult (parent or guardian).
        before_rows = df.shape[0]
        df = df[~((df["adults"] == 0) & ((df["children"] > 0) | (df["babies"] > 0)))]
        dropped = before_rows - df.shape[0]
        self.logger.info(f"Dropped {dropped} rows with illogical adults=0 but children/babies > 0")

        # Country
        # The missing values in the country column were replaced with "Unknown" because the
        # missingness itself appears to have a relationship with the target variable. The
        # cancellation rate for records with a missing country value is 13.73%, compared to
        # 37.03% for the overall dataset. Therefore, instead of removing these rows, the missing
        # values were treated as a separate category, "Unknown", to preserve potentially useful
        # information for the model.
        missing_country_count = df["country"].isna().sum()
        df["country"] = df["country"].fillna("Unknown")
        self.logger.info(f"Filled {missing_country_count} missing 'country' values with 'Unknown'")

        # Distribution channel
        before_rows = df.shape[0]
        df = df[df["distribution_channel"] != "Undefined"]
        dropped = before_rows - df.shape[0]
        self.logger.info(f"Dropped {dropped} rows with 'Undefined' distribution_channel")

        final_rows = df.shape[0]
        self.logger.info(
            f"Finished missing value handling. Rows before: {initial_rows}, "
            f"after: {final_rows}, total dropped: {initial_rows - final_rows}"
        )

        return df

    def create_new_features(self, df, info="train"):
        self.logger.info(f"[{info}] Starting creating new features")

        df = df.copy()
        self.logger.info(f"[{info}] Input dataframe shape: {df.shape}")

        # season
        def get_season(month):
            season_dict = {
                'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
                'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
                'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
                'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
            }
            return season_dict.get(month, 'Unknown')
        df['fe_season'] = df['arrival_date_month'].apply(get_season)
        self.logger.info(f"[{info}] Created 'fe_season' from 'arrival_date_month'")

        # total nights
        df["fe_total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        self.logger.info(f"[{info}] Created 'fe_total_nights' (stays_in_weekend_nights + stays_in_week_nights)")

        # is weekend stay
        df["fe_is_weekend_stay"] = (df["stays_in_weekend_nights"] > 0).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_is_weekend_stay' ({df['fe_is_weekend_stay'].sum()} rows with weekend stays)"
        )

        # weekend ratio
        df["fe_weekend_ratio"] = df["stays_in_weekend_nights"] / (df["fe_total_nights"] + 1)  # +1 is added to avoid division by zero.
        self.logger.info(f"[{info}] Created 'fe_weekend_ratio'")

        # total guests
        df["fe_total_guests"] = df["adults"] + df["children"] + df["babies"]
        self.logger.info(f"[{info}] Created 'fe_total_guests' (adults + children + babies)")

        # room type mismatch
        df["fe_room_type_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_room_type_mismatch' ({df['fe_room_type_mismatch'].sum()} rows with mismatched room types)"
        )

        # has previous cancellation
        df["fe_has_previous_cancellations"] = (df["previous_cancellations"] > 0).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_has_previous_cancellations' ({df['fe_has_previous_cancellations'].sum()} rows with prior cancellations)"
        )

        # has booking changes
        df["fe_has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_has_booking_changes' ({df['fe_has_booking_changes'].sum()} rows with booking changes)"
        )

        # deposit
        df["fe_has_deposit"] = (df["deposit_type"] != "No Deposit").astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_has_deposit' ({df['fe_has_deposit'].sum()} rows with a deposit)"
        )

        # car parking
        df["fe_has_parking_request"] = (df["required_car_parking_spaces"] > 0).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_has_parking_request' ({df['fe_has_parking_request'].sum()} rows requesting parking)"
        )

        # special requests
        df["fe_has_special_request"] = (df["total_of_special_requests"] > 0).astype(int)
        self.logger.info(
            f"[{info}] Created 'fe_has_special_request' ({df['fe_has_special_request'].sum()} rows with special requests)"
        )

        self.logger.info(f"[{info}] Finished creating new features. Output dataframe shape: {df.shape}")

        return df

    def handle_skewness(self, x_train, x_test, threshold=0.8, info="train/test"):
        self.logger.info(f"[{info}] Starting skewness handling (threshold={threshold})")

        x_train = x_train.copy()
        x_test = x_test.copy()

        # Identify numeric, continuous features (more than 2 unique values)
        numeric_features = x_train.select_dtypes(include="number").columns
        continuous_features = [
            col for col in numeric_features
            if x_train[col].nunique() > 2
        ]
        self.logger.info(f"[{info}] Found {len(continuous_features)} continuous numeric features")

        # Compute skewness on training data only
        skewness = x_train[continuous_features].skew()
        skewed_features = skewness[abs(skewness) >= threshold].sort_values(ascending=False)
        self.logger.info(
            f"[{info}] Found {len(skewed_features)} skewed features (|skew| >= {threshold}): "
            f"{skewed_features.index.tolist()}"
        )

        # Apply log1p transform only to non-negative skewed columns
        transformed_cols = []
        skipped_cols = []
        for col in skewed_features.index:
            if (x_train[col] >= 0).all():
                x_train[col] = np.log1p(x_train[col])
                x_test[col] = np.log1p(x_test[col])
                transformed_cols.append(col)
            else:
                skipped_cols.append(col)

        self.logger.info(f"[{info}] Applied log1p to {len(transformed_cols)} columns: {transformed_cols}")
        if skipped_cols:
            self.logger.info(
                f"[{info}] Skipped {len(skipped_cols)} columns due to negative values: {skipped_cols}"
            )

        # Log before/after comparison for transformed columns
        if transformed_cols:
            skewness_after = x_train[transformed_cols].skew()
            comparison = pd.DataFrame({
                "before": skewness[transformed_cols],
                "after": skewness_after
            }).sort_values("before", ascending=False)
            self.logger.info(f"[{info}] Skewness before/after transformation:\n{comparison}")

        self.logger.info(f"[{info}] Finished skewness handling")

        return x_train, x_test

    def encode_train(self, df, info="train"):
        self.logger.info(f"[{info}] Starting label encoding")

        df = df.copy()
        encoders = {}

        categorical_cols = df.select_dtypes(exclude=np.number).columns
        self.logger.info(f"[{info}] Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            self.logger.info(f"[{info}] Encoded '{col}' ({len(le.classes_)} unique categories)")

        self.logger.info(f"[{info}] Finished label encoding. Output shape: {df.shape}")

        return df, encoders

    def encode_test(self, df, encoders, train_columns, info="test"):
        self.logger.info(f"[{info}] Starting label encoding using fitted encoders")

        df = df.copy()

        for col, le in encoders.items():
            df[col] = df[col].astype(str)
            unseen_mask = ~df[col].isin(le.classes_)
            unseen_count = unseen_mask.sum()
            if unseen_count > 0:
                self.logger.info(f"[{info}] Column '{col}' has {unseen_count} unseen categories (mapped to -1)")

            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        before_cols = df.shape[1]
        df = df.reindex(columns=train_columns, fill_value=0)
        self.logger.info(
            f"[{info}] Aligned columns to match training set ({before_cols} -> {df.shape[1]} columns)"
        )

        self.logger.info(f"[{info}] Finished label encoding. Output shape: {df.shape}")

        return df

    def scale_train(self, df, info="train"):
        self.logger.info(f"[{info}] Starting feature scaling")

        df = df.copy()
        scalers = {}

        numeric_cols = df.select_dtypes(include=np.number).columns
        self.logger.info(f"[{info}] Found {len(numeric_cols)} numeric columns to scale")

        for col in numeric_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

        self.logger.info(f"[{info}] Finished feature scaling. Output shape: {df.shape}")

        return df, scalers

    def scale_test(self, df, scalers, info="test"):
        self.logger.info(f"[{info}] Starting feature scaling using fitted scalers")

        df = df.copy()

        scaled_count = 0
        for col in df.columns:
            if col in scalers:
                df[col] = scalers[col].transform(df[[col]])
                scaled_count += 1

        self.logger.info(f"[{info}] Scaled {scaled_count} columns")
        self.logger.info(f"[{info}] Finished feature scaling. Output shape: {df.shape}")

        return df
    
    def save_data(self, x_train, x_test, y_train, y_test):
        train_df = x_train.copy()
        train_df["is_canceled"] = y_train.values

        test_df = x_test.copy()
        test_df["is_canceled"] = y_test.values

        self.logger.info(f"Training dataset shape: {train_df.shape}")
        self.logger.info(f"Testing dataset shape: {test_df.shape}")

        self.logger.info(f"Output directory ready: {self.output_dir_path}")

        # Save datasets
        train_path = self.output_dir_path / "train_preprocessed.csv"
        test_path = self.output_dir_path / "test_preprocessed.csv"

        self.logger.info("Saving preprocessed training dataset...")
        train_df.to_csv(train_path, index=False)

        self.logger.info("Saving preprocessed testing dataset...")
        test_df.to_csv(test_path, index=False)

        self.logger.info(f"Training dataset saved to: {train_path}")
        self.logger.info(f"Testing dataset saved to: {test_path}")
        self.logger.info("Preprocessed datasets saved successfully.")

    def save_artifacts(self, encoders, scalers, train_columns, threshold=None):
        """
        Saves the fitted encoders, scalers, and training column order so they
        can be reused later for inference on new data.
        """
        self.logger.info("Starting to save preprocessing artifacts...")

        artifacts_dir = ROOT_DIR / "models" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)


        encoders_path = artifacts_dir / "encoders.pkl"
        scalers_path = artifacts_dir / "scalers.pkl"
        columns_path = artifacts_dir / "train_columns.pkl"

        joblib.dump(encoders, encoders_path)
        self.logger.info(f"Encoders saved to: {encoders_path}")

        joblib.dump(scalers, scalers_path)
        self.logger.info(f"Scalers saved to: {scalers_path}")

        joblib.dump(list(train_columns), columns_path)
        self.logger.info(f"Train columns saved to: {columns_path}")

        if threshold is not None:
            threshold_path = artifacts_dir / "threshold.pkl"
            joblib.dump(threshold, threshold_path)
            self.logger.info(f"Threshold saved to: {threshold_path}")

        self.logger.info("All preprocessing artifacts saved successfully.")