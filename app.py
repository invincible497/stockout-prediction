import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df
def forecast_stock(data, weeks=12, stock_threshold=150):
    # Ensure 'Sales' column is numeric
    data = data.dropna()  # Remove missing values
    data = data.astype(float)  # Convert to numeric
    
    # Ensure Date is the index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Error: The index must be a Date column in datetime format.")

    # Train ARIMA Model
    model = ARIMA(data, order=(2, 0, 2))
    result = model.fit()

    # Forecast future sales
    future_forecast = result.forecast(steps=weeks)
    future_dates = pd.date_range(start=data.index[-1], periods=weeks + 1, freq='W')[1:]
    
    # Create DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': future_forecast})
    forecast_df['Stockout Risk'] = forecast_df['Forecasted_Sales'] < stock_threshold

    return forecast_df


st.title("Medicine Stockout Forecast App")
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Historical Sales Data")
    st.line_chart(data)
    
    forecast_df = forecast_stock(data)
    st.write("### Forecasted Sales for Next 12 Weeks")
    st.dataframe(forecast_df)
    
    # Plot forecast
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data, label="Historical Sales", color="blue")
    plt.plot(forecast_df['Date'], forecast_df['Forecasted_Sales'], label="Forecast", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    st.pyplot(plt)
    
    st.write("### Stockout Risk Alerts")
    stockout_weeks = forecast_df[forecast_df['Stockout Risk']]
    if not stockout_weeks.empty:
        st.warning("Stockout risk detected in the following weeks:")
        st.dataframe(stockout_weeks)
    else:
        st.success("No stockout risk detected!")
