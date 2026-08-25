import requests
import gradio as gr


API_URL = "http://127.0.0.1:8000/predict"


def predict(
    hotel,
    lead_time,
    arrival_date_month,
    arrival_date_week_number,
    arrival_date_day_of_month,
    stays_in_weekend_nights,
    stays_in_week_nights,
    adults,
    children,
    babies,
    meal,
    country,
    market_segment,
    distribution_channel,
    is_repeated_guest,
    previous_cancellations,
    previous_bookings_not_canceled,
    reserved_room_type,
    assigned_room_type,
    booking_changes,
    deposit_type,
    days_in_waiting_list,
    customer_type,
    adr,
    required_car_parking_spaces,
    total_of_special_requests,
    city
):

    data = {
        "hotel": hotel,
        "lead_time": int(lead_time),
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": int(arrival_date_week_number),
        "arrival_date_day_of_month": int(arrival_date_day_of_month),
        "stays_in_weekend_nights": int(stays_in_weekend_nights),
        "stays_in_week_nights": int(stays_in_week_nights),
        "adults": int(adults),
        "children": float(children),
        "babies": int(babies),
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": int(is_repeated_guest),
        "previous_cancellations": int(previous_cancellations),
        "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": int(booking_changes),
        "deposit_type": deposit_type,
        "days_in_waiting_list": int(days_in_waiting_list),
        "customer_type": customer_type,
        "adr": float(adr),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
        "city": city
    }

    try:

        response = requests.post(
            API_URL,
            json=data,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        prediction = result["prediction"]
        label = result["prediction_label"]
        probability = result["probability"]
        threshold = result["threshold"]

        return (
            f"Prediction: {label}\n"
            f"Cancellation Probability: {probability:.2%}\n"
            f"Threshold: {threshold:.2f}"
        )

    except requests.exceptions.ConnectionError:

        return (
            "Error: FastAPI server is not running.\n"
            "Please start the FastAPI server first."
        )

    except requests.exceptions.HTTPError as e:

        return f"API Error: {e}\n{response.text}"

    except Exception as e:

        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=predict,

    inputs=[
        gr.Dropdown(
            ["City Hotel", "Resort Hotel"],
            label="Hotel",
            value="City Hotel"
        ),

        gr.Number(
            label="Lead Time",
            value=45
        ),

        gr.Dropdown(
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ],
            label="Arrival Month",
            value="July"
        ),

        gr.Number(
            label="Arrival Week Number",
            value=27
        ),

        gr.Number(
            label="Arrival Day of Month",
            value=15
        ),

        gr.Number(
            label="Weekend Nights",
            value=1
        ),

        gr.Number(
            label="Week Nights",
            value=2
        ),

        gr.Number(
            label="Adults",
            value=2
        ),

        gr.Number(
            label="Children",
            value=0
        ),

        gr.Number(
            label="Babies",
            value=0
        ),

        gr.Dropdown(
            ["BB", "HB", "FB", "SC", "Undefined"],
            label="Meal",
            value="BB"
        ),

        gr.Textbox(
            label="Country",
            value="PRT"
        ),

        gr.Dropdown(
            [
                "Online TA",
                "Offline TA/TO",
                "Direct",
                "Groups",
                "Corporate",
                "Complementary",
                "Aviation",
                "Undefined"
            ],
            label="Market Segment",
            value="Online TA"
        ),

        gr.Dropdown(
            [
                "TA/TO",
                "Direct",
                "Corporate",
                "GDS",
                "Undefined"
            ],
            label="Distribution Channel",
            value="TA/TO"
        ),

        gr.Radio(
            [0, 1],
            label="Repeated Guest",
            value=0
        ),

        gr.Number(
            label="Previous Cancellations",
            value=0
        ),

        gr.Number(
            label="Previous Bookings Not Canceled",
            value=0
        ),

        gr.Textbox(
            label="Reserved Room Type",
            value="A"
        ),

        gr.Textbox(
            label="Assigned Room Type",
            value="A"
        ),

        gr.Number(
            label="Booking Changes",
            value=0
        ),

        gr.Dropdown(
            [
                "No Deposit",
                "Non Refund",
                "Refundable"
            ],
            label="Deposit Type",
            value="No Deposit"
        ),

        gr.Number(
            label="Days in Waiting List",
            value=0
        ),

        gr.Dropdown(
            [
                "Transient",
                "Transient-Party",
                "Contract",
                "Group"
            ],
            label="Customer Type",
            value="Transient"
        ),

        gr.Number(
            label="ADR",
            value=100.0
        ),

        gr.Number(
            label="Required Car Parking Spaces",
            value=0
        ),

        gr.Number(
            label="Total Special Requests",
            value=0
        ),

        gr.Textbox(
            label="City",
            value="Lisbon"
        )
    ],

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="🏨 Hotel Booking Cancellation Prediction",

    description=(
        "Enter hotel booking information to predict "
        "whether the booking is likely to be canceled."
    )
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )