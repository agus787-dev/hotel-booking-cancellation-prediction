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
        # threshold = result["threshold"]

        return (
            f"Prediction: {label}\n"
            f"Cancellation Probability: {probability:.2%}\n"
            # f"Threshold: {threshold:.2f}"
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


import gradio as gr


custom_css = """
/* Main background */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: #f6f8fb !important;
}

/* Hero */
.hero {
    padding: 35px 40px;
    border-radius: 20px;
    margin-bottom: 25px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
}

.hero h1 {
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 16px;
    opacity: 0.85;
}

/* Section cards */
.section-card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Predict button */
.predict-btn {
    height: 60px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
}

/* Result */
.result-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    border: 1px solid #e5e7eb;
}

/* Footer */
.footer {
    text-align: center;
    padding: 25px;
    color: #6b7280;
    font-size: 13px;
}
"""


with gr.Blocks(
    title="Hotel Cancellation Predictor",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    )
) as demo:

    # =========================
    # HERO
    # =========================

    gr.HTML("""
    <div class="hero">
        <h1>🏨 Hotel Cancellation Predictor</h1>
        <p>
            Predict the probability of a hotel booking cancellation
            using a machine learning model.
        </p>
    </div>
    """)

    # =========================
    # BOOKING DETAILS
    # =========================

    with gr.Group(elem_classes="section-card"):

        gr.HTML("""
        <div class="section-title">
            📅 Booking Details
        </div>
        """)

        with gr.Row():

            hotel = gr.Dropdown(
                ["City Hotel", "Resort Hotel"],
                label="Hotel",
                value="City Hotel"
            )

            lead_time = gr.Number(
                label="Lead Time (days)",
                value=45
            )

            arrival_date_month = gr.Dropdown(
                [
                    "January", "February", "March", "April",
                    "May", "June", "July", "August",
                    "September", "October", "November", "December"
                ],
                label="Arrival Month",
                value="July"
            )

        with gr.Row():

            arrival_date_week_number = gr.Number(
                label="Arrival Week",
                value=27
            )

            arrival_date_day_of_month = gr.Number(
                label="Arrival Day",
                value=15
            )

            stays_in_weekend_nights = gr.Number(
                label="Weekend Nights",
                value=1
            )

            stays_in_week_nights = gr.Number(
                label="Week Nights",
                value=2
            )

    # =========================
    # GUEST INFORMATION
    # =========================

    with gr.Group(elem_classes="section-card"):

        gr.HTML("""
        <div class="section-title">
            👤 Guest Information
        </div>
        """)

        with gr.Row():

            adults = gr.Number(
                label="Adults",
                value=2
            )

            children = gr.Number(
                label="Children",
                value=0
            )

            babies = gr.Number(
                label="Babies",
                value=0
            )

            is_repeated_guest = gr.Radio(
                [0, 1],
                label="Repeated Guest",
                value=0
            )

        with gr.Row():

            country = gr.Textbox(
                label="Country",
                value="PRT"
            )

            city = gr.Textbox(
                label="City",
                value="Lisbon"
            )

            customer_type = gr.Dropdown(
                [
                    "Transient",
                    "Transient-Party",
                    "Contract",
                    "Group"
                ],
                label="Customer Type",
                value="Transient"
            )

    # =========================
    # BOOKING CHANNEL
    # =========================

    with gr.Group(elem_classes="section-card"):

        gr.HTML("""
        <div class="section-title">
            📡 Booking Channel
        </div>
        """)

        with gr.Row():

            market_segment = gr.Dropdown(
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
            )

            distribution_channel = gr.Dropdown(
                [
                    "TA/TO",
                    "Direct",
                    "Corporate",
                    "GDS",
                    "Undefined"
                ],
                label="Distribution Channel",
                value="TA/TO"
            )

            meal = gr.Dropdown(
                [
                    "BB",
                    "HB",
                    "FB",
                    "SC",
                    "Undefined"
                ],
                label="Meal Plan",
                value="BB"
            )

    # =========================
    # ROOM & PAYMENT
    # =========================

    with gr.Group(elem_classes="section-card"):

        gr.HTML("""
        <div class="section-title">
            🛏️ Room & Payment
        </div>
        """)

        with gr.Row():

            reserved_room_type = gr.Textbox(
                label="Reserved Room Type",
                value="A"
            )

            assigned_room_type = gr.Textbox(
                label="Assigned Room Type",
                value="A"
            )

            deposit_type = gr.Dropdown(
                [
                    "No Deposit",
                    "Non Refund",
                    "Refundable"
                ],
                label="Deposit Type",
                value="No Deposit"
            )

            adr = gr.Number(
                label="ADR (€)",
                value=100.0
            )

        with gr.Row():

            booking_changes = gr.Number(
                label="Booking Changes",
                value=0
            )

            days_in_waiting_list = gr.Number(
                label="Waiting List Days",
                value=0
            )

            required_car_parking_spaces = gr.Number(
                label="Parking Spaces",
                value=0
            )

            total_of_special_requests = gr.Number(
                label="Special Requests",
                value=0
            )

    # =========================
    # BOOKING HISTORY
    # =========================

    with gr.Group(elem_classes="section-card"):

        gr.HTML("""
        <div class="section-title">
            📊 Booking History
        </div>
        """)

        with gr.Row():

            previous_cancellations = gr.Number(
                label="Previous Cancellations",
                value=0
            )

            previous_bookings_not_canceled = gr.Number(
                label="Previous Bookings Not Canceled",
                value=0
            )

    # =========================
    # PREDICT BUTTON
    # =========================

    predict_button = gr.Button(
        "🎯 Predict Cancellation Risk",
        variant="primary",
        elem_classes="predict-btn"
    )

    # =========================
    # RESULT
    # =========================

    with gr.Group(elem_classes="result-card"):

        gr.HTML("""
        <div class="section-title">
            🔮 Prediction Result
        </div>
        """)

        output = gr.Textbox(
            label="",
            placeholder="Your prediction will appear here...",
            lines=5
        )

    # =========================
    # BUTTON FUNCTION
    # =========================

    predict_button.click(
        fn=predict,
        inputs=[
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
        ],
        outputs=output
    )

    # =========================
    # FOOTER
    # =========================

    gr.HTML("""
    <div class="footer">
        Built with FastAPI • Gradio • Machine Learning
    </div>
    """)


# demo = gr.Interface(
#     fn=predict,

#     inputs=[
#         gr.Dropdown(
#             ["City Hotel", "Resort Hotel"],
#             label="Hotel",
#             value="City Hotel"
#         ),

#         gr.Number(
#             label="Lead Time",
#             value=45
#         ),

#         gr.Dropdown(
#             [
#                 "January", "February", "March", "April",
#                 "May", "June", "July", "August",
#                 "September", "October", "November", "December"
#             ],
#             label="Arrival Month",
#             value="July"
#         ),

#         gr.Number(
#             label="Arrival Week Number",
#             value=27
#         ),

#         gr.Number(
#             label="Arrival Day of Month",
#             value=15
#         ),

#         gr.Number(
#             label="Weekend Nights",
#             value=1
#         ),

#         gr.Number(
#             label="Week Nights",
#             value=2
#         ),

#         gr.Number(
#             label="Adults",
#             value=2
#         ),

#         gr.Number(
#             label="Children",
#             value=0
#         ),

#         gr.Number(
#             label="Babies",
#             value=0
#         ),

#         gr.Dropdown(
#             ["BB", "HB", "FB", "SC", "Undefined"],
#             label="Meal",
#             value="BB"
#         ),

#         gr.Textbox(
#             label="Country",
#             value="PRT"
#         ),

#         gr.Dropdown(
#             [
#                 "Online TA",
#                 "Offline TA/TO",
#                 "Direct",
#                 "Groups",
#                 "Corporate",
#                 "Complementary",
#                 "Aviation",
#                 "Undefined"
#             ],
#             label="Market Segment",
#             value="Online TA"
#         ),

#         gr.Dropdown(
#             [
#                 "TA/TO",
#                 "Direct",
#                 "Corporate",
#                 "GDS",
#                 "Undefined"
#             ],
#             label="Distribution Channel",
#             value="TA/TO"
#         ),

#         gr.Radio(
#             [0, 1],
#             label="Repeated Guest",
#             value=0
#         ),

#         gr.Number(
#             label="Previous Cancellations",
#             value=0
#         ),

#         gr.Number(
#             label="Previous Bookings Not Canceled",
#             value=0
#         ),

#         gr.Textbox(
#             label="Reserved Room Type",
#             value="A"
#         ),

#         gr.Textbox(
#             label="Assigned Room Type",
#             value="A"
#         ),

#         gr.Number(
#             label="Booking Changes",
#             value=0
#         ),

#         gr.Dropdown(
#             [
#                 "No Deposit",
#                 "Non Refund",
#                 "Refundable"
#             ],
#             label="Deposit Type",
#             value="No Deposit"
#         ),

#         gr.Number(
#             label="Days in Waiting List",
#             value=0
#         ),

#         gr.Dropdown(
#             [
#                 "Transient",
#                 "Transient-Party",
#                 "Contract",
#                 "Group"
#             ],
#             label="Customer Type",
#             value="Transient"
#         ),

#         gr.Number(
#             label="ADR",
#             value=100.0
#         ),

#         gr.Number(
#             label="Required Car Parking Spaces",
#             value=0
#         ),

#         gr.Number(
#             label="Total Special Requests",
#             value=0
#         ),

#         gr.Textbox(
#             label="City",
#             value="Lisbon"
#         )
#     ],

#     outputs=gr.Textbox(
#         label="Prediction Result"
#     ),

#     title="🏨 Hotel Booking Cancellation Prediction",

#     description=(
#         "Enter hotel booking information to predict "
#         "whether the booking is likely to be canceled."
#     )
# )


# if __name__ == "__main__":
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860
#     )

    