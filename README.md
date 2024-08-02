# SALES FORECASTING


## BUSINESS REQUIREMENT
To predict future sales quantities and provide insightful visualizations to assist in strategic decision-making and Inventory Management.

### Scope:
    1. Sales Forecasting:
   - Utilize historical sales data to forecast future sales.
   - Implement multiple forecasting models and finalize the best performing one.
   - Provide a user-friendly interface for business stakeholders to interact with the forecasts.

    2. Sales Dashboard:
   - Visualize historical sales data and forecasted sales.
   - Provide key performance indicators (KPIs) to monitor sales performance.
   - Enable regional sales analysis to identify high-performing and low-performing regions.

### Methodology

    1. Data Collection:
   - Gather historical sales data (daily sales quantities) from the database or CSV files.

    2. Data Preprocessing:
   - Clean and preprocess the data to handle missing values, outliers, and seasonality.

    3. Modeling:
   - Implement multiple forecasting models (e.g., ARIMA, Exponential Smoothing, Prophet, Simple Moving Averages (SMA)).
   - Compare model performance using metrics like RMSE, MAE, and MAPE.
   - Finalize the SMA model based on its performance.

    4. Deployment:
   - Create a Streamlit application to allow users to upload their sales data, select the forecasting period, and view forecasted results.
   - Provide an option to download the forecasted data as a CSV file.

    5. Visualization:
   - Use Tableau to create an interactive sales dashboard.
   - Visualize key metrics and trends to provide actionable insights.

### Charts and KPIs for the Dashboard

    1. Sales Trend Over Time:
   - Line chart showing historical sales data and forecasted sales for the selected period.
   - Helps identify trends, seasonality, and patterns in sales.

    2. Total Sales Quantity:
   - Pie chart showing the distribution of total sales quantity over different periods or categories.
   - Provides a quick overview of sales distribution.

    3. Region-wise Sales:
   - Map chart showing sales distribution across different regions.
   - Bar chart showing sales quantities for each region.
   - Helps identify high-performing and low-performing regions.

    4. Total Profit:
   - Bubble chart showing profit distribution across different categories.
   - Helps understand profitability in various segments.

    5. Key Performance Indicators (KPIs):
   - Total Sales: The total sales amount in dollars.
   - Total Profit: The total profit in dollars.
   - Total Customers: The total number of unique customers.
   - Sales Growth: Percentage increase or decrease in sales over the previous period.
   - Profit Margin: The profit as a percentage of total sales.

### Additional Information

    1. Technical Stack:
   - Data Analysis and Modeling: Python (Pandas, NumPy, scikit-learn, statsmodels)
   - Forecasting Model: Simple Moving Averages (SMA)
   - Web Application: Streamlit
   - Visualization: Tableau

    2. Usage Instructions:
   - Streamlit Application:
     - Upload the sales data CSV file.
     - Select the forecasting period (30, 60, 90, 120 days).
     - View the forecasted sales data and total forecasted quantity.
     - Download the forecasted data as a CSV file.

### Tableau Dashboard:
     - Navigate through the dashboard to view different visualizations.
     - Use filters to focus on specific regions or time periods.
     - Analyze the KPIs for strategic insights.


# Forecasting Overview (Streamlit)

![STREAMLIT Overview](https://github.com/user-attachments/assets/8c79d29d-25c1-4956-81ea-86ae387fd392)

# Streamlit link
http://localhost:8501/#forecasting-product-quantity

# Report Snapshot (Tableau)

![Sales Dashboard](https://github.com/user-attachments/assets/6473fcba-8139-4efc-aaf5-f59abe2c1553)
