import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle

# Function to calculate Simple Moving Average
def simple_moving_average(data, window):
    history = data[-window:]
    return np.mean(history)

# Helper function to create a download link for a DataFrame as CSV
def get_binary_file_downloader_html(df, title, button_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{title}">{button_text}</a>'
    return href

# Load the Simple Moving Average model from pickle
with open('sma_model.pkl', 'rb') as f:
    sma_model = pickle.load(f)


# Display a big heading for the forecasting section
st.markdown('<h1 style="text-align: center;">Forecasting Product Quantity</h1>', unsafe_allow_html=True)

# Allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load your dataset into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Assuming 'date' and 'Quantity' columns exist in your dataset
    # Sort the dataset by the "date" column (if it's not already sorted)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Extract the "Quantity" values
    data = df['Quantity'].values

    # Define the forecasting periods
    periods = [30, 60, 90, 120]

    # Display buttons for different forecasting periods with customized appearance
    st.subheader("Select forecasting period:")
    button_col1, button_col2, button_col3, button_col4 = st.columns(4)
    
    if button_col1.button("30 Days", key="30_days"):
        selected_period = 30
    if button_col2.button("60 Days", key="60_days"):
        selected_period = 60
    if button_col3.button("90 Days", key="90_days"):
        selected_period = 90
    if button_col4.button("120 Days", key="120_days"):
        selected_period = 120

    # Forecast for the selected period using Simple Moving Average
    if 'selected_period' in locals():
        forecast = []
        window = 7  # Adjust this based on your model's requirements

        for i in range(selected_period):
            next_day_forecast = simple_moving_average(data, window)
            forecast.append(next_day_forecast)
            data = np.append(data, next_day_forecast)

        # Create a date range for the forecasted period
        last_date = df['date'].max()
        forecasted_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=selected_period, freq='D')

        # Create a DataFrame for the forecasted values
        forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Quantity': forecast})

        # Convert Quantity to integer (if needed)
        forecasted_df['Quantity'] = forecasted_df['Quantity'].astype(int)

 # Calculate the sum of the forecasted quantities
        total_forecasted_quantity = forecasted_df['Quantity'].sum()

        # Display forecasted data and the sum in Streamlit
        st.subheader(f'Forecasted Data for {selected_period} days:')
        st.write(forecasted_df)
        st.subheader(f'Total Forecasted Quantity for {selected_period} days: {total_forecasted_quantity}')


        # Download forecasted data as CSV with a rectangular-shaped button
        st.subheader('Download Forecasted Data:')
        st.write('Click the button below to download the forecasted data as a CSV file.')

        # Function to generate download link
        def download_link(df, title, button_text):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{title}">{button_text}</a>'
            return href

        # Display download button
        download_button_text = f'Download {selected_period} Days Forecast Data'
        download_link_str = download_link(forecasted_df, f'SMA_forecast_{selected_period}days.csv', download_button_text)
        st.markdown(download_link_str, unsafe_allow_html=True)
