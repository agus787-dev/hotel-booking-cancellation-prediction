from pathlib import Path
import sys
import pandas as pd

sys.path.append("..")
ROOT_DIR = Path(__file__).resolve().parent.parent

from src.inference_pipeline import InferencePipeline

if __name__ == "__main__":
    pipeline = InferencePipeline(log_file="offline_check.log")

    # --- Option 1: test a single booking manually ---
    sample_booking = {
        "hotel": "City Hotel",
        "lead_time": 45,
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
        "city": "Lisbon"
    }

    result = pipeline.predict_one(sample_booking)
    print("\n--- Single Prediction ---")
    print(f"Prediction: {'Canceled' if result['prediction'] == 1 else 'Not Canceled'}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Threshold used: {result['threshold']:.4f}")

    # --- Option 2: test a batch of new bookings from a CSV ---
    # new_data_path = ROOT_DIR / "data" / "raw" / "new_bookings_to_check.csv"
    # new_df = pd.read_csv(new_data_path)
    # results_df = pipeline.predict_batch(new_df)
    # print("\n--- Batch Predictions ---")
    # print(results_df[["prediction", "cancellation_probability"]].head(10))
    # results_df.to_csv(ROOT_DIR / "data" / "outputs" / "batch_predictions.csv", index=False)