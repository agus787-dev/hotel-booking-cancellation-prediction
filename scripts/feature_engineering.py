
import pandas as pd
import numpy as np
import sys
import os

sys.path.append("..")
from src.feature_engineering import FeatureEngineer
from src.data_loader import DataLoader


loader=DataLoader("../data/raw/hotel_bookings_updated_2024.csv", "feature_engineering")
fe = FeatureEngineer("feature_engineering")

# Loading data
df = loader.load()

# drop and handling missing values
new_df = fe.drop_and_handling_missing_values(df)

# Splitting
x_train, x_test, y_train, y_test = loader.split(new_df, target_columns=["is_canceled"])

# Creating new features
x_train_fe = fe.create_new_features(x_train, "train dataset")
x_test_fe = fe.create_new_features(x_test, "test dataset")

# Skewness
x_train_transformed, x_test_transformed = fe.handle_skewness(x_train_fe, x_test_fe, info="train/test")

# Encoding
x_train_encoded, encoders = fe.encode_train(x_train_transformed, info="train")
x_test_encoded = fe.encode_test(x_test_transformed, encoders, x_train_encoded.columns, info="test")

# Scaling
x_train_scaled, scalers = fe.scale_train(x_train_encoded, info="train")
x_test_scaled = fe.scale_test(x_test_encoded, scalers, info="test")

# Saving dataset
fe.save_data(x_train_scaled, x_test_scaled, y_train, y_test)

# Save encoders, scalers, and column order (threshold added later, after model training)
fe.save_artifacts(encoders, scalers, x_train_scaled.columns)






