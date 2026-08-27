import sys
from pathlib import Path

sys.path.append("..")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

from src.inference_pipeline import InferencePipeline
from demo.app import demo


app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description="Predicts whether a hotel booking is likely to be canceled",
    version="1.0.0"
)

pipeline = InferencePipeline(log_file="api.log")


class BookingInput(BaseModel):
    hotel: str = Field(..., example="City Hotel")
    lead_time: int = Field(..., example=45)
    arrival_date_month: str = Field(..., example="July")
    arrival_date_week_number: int = Field(..., example=27)
    arrival_date_day_of_month: int = Field(..., example=15)
    stays_in_weekend_nights: int = Field(..., example=1)
    stays_in_week_nights: int = Field(..., example=2)
    adults: int = Field(..., example=2)
    children: float = Field(..., example=0)
    babies: int = Field(..., example=0)
    meal: str = Field(..., example="BB")
    country: str = Field(..., example="PRT")
    market_segment: str = Field(..., example="Online TA")
    distribution_channel: str = Field(..., example="TA/TO")
    is_repeated_guest: int = Field(..., example=0)
    previous_cancellations: int = Field(..., example=0)
    previous_bookings_not_canceled: int = Field(..., example=0)
    reserved_room_type: str = Field(..., example="A")
    assigned_room_type: str = Field(..., example="A")
    booking_changes: int = Field(..., example=0)
    deposit_type: str = Field(..., example="No Deposit")
    days_in_waiting_list: int = Field(..., example=0)
    customer_type: str = Field(..., example="Transient")
    adr: float = Field(..., example=100.0)
    required_car_parking_spaces: int = Field(..., example=0)
    total_of_special_requests: int = Field(..., example=0)
    city: str = Field(..., example="Lisbon")


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    threshold: float


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Hotel Booking Cancellation Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(booking: BookingInput):
    try:
        raw_input = booking.model_dump()
        result = pipeline.predict_one(raw_input)

        return PredictionOutput(
            prediction=result["prediction"],
            prediction_label="Canceled" if result["prediction"] == 1 else "Not Canceled",
            probability=result["probability"],
            threshold=result["threshold"]
        )

    except Exception as e:
        pipeline.logger.info(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )