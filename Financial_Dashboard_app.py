import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import request

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .css-15zrgzn {
        background-color: #f5f5f5 !important;
    }
    .stSelectbox > div {
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title of the dashboard
st.title("📈 Yahoo Financial Insight Dashboard")

# Function to get S&P 500 companies
@st.cache_data(ttl=3600)
def get_sp500_companies():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]['Symbol'].tolist()

# Sidebar for user input
st.sidebar.header("📋 User Input")
sp500_companies = get_sp500_companies()
stock_symbol = st.sidebar.selectbox(
    "Select Stock Symbol (S&P 500):",
    sp500_companies,
    index=sp500_companies.index("AAPL"),
)

# Fetching stock data
if stock_symbol:
    @st.cache_data(ttl=3600)
    def load_data(symbol):
        try:
            return yf.download(symbol, start="1900-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            st.error(traceback.format_exc())
            return pd.DataFrame()

    if st.sidebar.button("🔄 Update Data"):
        stock_data = load_data(stock_symbol)
    else:
        stock_data = load_data(stock_symbol)

    # Fetch additional stock information
    stock_info = yf.Ticker(stock_symbol).info

    # Create tabs
    tabs = st.tabs(
        [
            "📊 Summary",
            "📈 Chart",
            "📑 Financials",
            "🎲 Monte Carlo Simulation",
            "📉 Analytics",
        ]
    )

    # Function to filter data by duration
    def filter_data_by_duration(data, duration):
        if duration == "1M":
            return data.last("1M")
        elif duration == "3M":
            return data.last("3M")
        elif duration == "6M":
            return data.last("6M")
        elif duration == "YTD":
            return data[data.index.year == pd.Timestamp.today().year]
        elif duration == "1Y":
            return data.last("1Y")
        elif duration == "3Y":
            return data.last("3Y")
        elif duration == "5Y":
            return data.last("5Y")
        elif duration == "MAX":
            return data
        else:
            return data

    # Summary Tab
    with tabs[0]:
        st.subheader(f"🔍 {stock_symbol} Summary")
        st.markdown("---")

        # Display stock information
        st.markdown(f"**📈 Previous Close:** {stock_info.get('previousClose', 'N/A')}")
        st.markdown(f"**📊 Open:** {stock_info.get('open', 'N/A')}")
        st.markdown(f"**📉 Bid:** {stock_info.get('bid', 'N/A')} x {stock_info.get('bidSize', 'N/A')}")
        st.markdown(f"**💹 Day's Range:** {stock_info.get('dayLow', 'N/A')} - {stock_info.get('dayHigh', 'N/A')}")

        st.markdown("---")

        # Add company profile
        st.subheader("🏢 Company Profile")
        st.markdown(stock_info.get("longBusinessSummary", "N/A"))

        # Duration Selection
        st.subheader("📅 Filter Data")
        duration_options = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
        duration = st.selectbox(
            "Select Duration:", duration_options, index=duration_options.index("1Y"), key="select_summary"
        )

        # Filter data
        chart_data = filter_data_by_duration(stock_data, duration)

        # Plot Summary
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        area_plot = go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            fill="tozeroy",
            fillcolor="rgba(60, 179, 113, 0.2)",
            line=dict(color="green", width=2),
            name="Stock Price",
        )
        fig.add_trace(area_plot, secondary_y=True)

        bar_plot = go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            marker=dict(color="rgba(30, 144, 255, 0.6)"),
            name="Volume",
        )
        fig.add_trace(bar_plot, secondary_y=False)

        # Update layout
        fig.update_layout(
            title=f"{stock_symbol} Price & Volume",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            template="plotly_white",
        )
        st.plotly_chart(fig)

    # Chart Tab
    with tabs[1]:
        st.subheader(f"📈 {stock_symbol} Chart")
        st.markdown("---")

        # Date Range Selection
        predefined_ranges = ["Last 30 days", "Last 60 days", "Last 90 days", "Custom"]
        selected_range = st.selectbox("Select Date Range:", predefined_ranges)

        if selected_range == "Custom":
            start_date = st.date_input("Start Date", value=pd.Timestamp.today() - pd.DateOffset(years=1))
            end_date = st.date_input("End Date", value=pd.Timestamp.today())
        else:
            days = int(selected_range.split()[1])
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(days=days)

        # Duration Selection
        duration_options = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
        duration = st.selectbox(
            "Select Duration:", duration_options, index=duration_options.index("1Y"), key="select_duration-chart"
        )

        # Time Interval Selection
        interval_options = ["1d", "1mo", "1y"]
        interval = st.selectbox("Select Time Interval:", interval_options, index=interval_options.index("1d"), key="select_interval_chart")

        # Plot Type Selection
        plot_type = st.selectbox("Select Plot Type:", ["Line", "Candle"], key="select_plot_chart")

        # Filter data based on selected duration
        chart_data = filter_data_by_duration(stock_data, duration)

        # Filter data based on selected date range
        chart_data = chart_data[(chart_data.index >= pd.to_datetime(start_date)) & (chart_data.index <= pd.to_datetime(end_date))]

        # Calculate Simple Moving Average (MA)
        chart_data["MA50"] = chart_data["Close"].rolling(window=50).mean()

        # Plotting
        fig = go.Figure()

        if plot_type == "Line":
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Close Price"))
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MA50"], mode="lines", name="MA50"))
        elif plot_type == "Candle":
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data["Open"],
                    high=chart_data["High"],
                    low=chart_data["Low"],
                    close=chart_data["Close"],
                    name="Candlestick",
                )
            )
            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["MA50"], mode="lines", name="MA50"))

        # Volume Plot
        fig.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", yaxis="y2", opacity=0.2))

        # Layout
        fig.update_layout(
            title=f"{stock_symbol} Stock Price",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right"),
            xaxis=dict(title="Date"),
            legend=dict(x=0, y=1),
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(fig)

    # Financials Tab
    with tabs[2]:
        st.subheader(f"📑 {stock_symbol} Financials")
        st.markdown("---")

        # Financial Statement Selection
        financial_statement_options = ["Income Statement", "Balance Sheet", "Cash Flow"]
        financial_statement = st.selectbox("Select Financial Statement:", financial_statement_options)

        # Period Selection
        period_options = ["Annual", "Quarterly"]
        period = st.selectbox("Select Period:", period_options)

        # Fetch Financial Data
        if period == "Annual":
            if financial_statement == "Income Statement":
                financial_data = yf.Ticker(stock_symbol).financials
            elif financial_statement == "Balance Sheet":
                financial_data = yf.Ticker(stock_symbol).balance_sheet
            elif financial_statement == "Cash Flow":
                financial_data = yf.Ticker(stock_symbol).cashflow
        elif period == "Quarterly":
            if financial_statement == "Income Statement":
                financial_data = yf.Ticker(stock_symbol).quarterly_financials
            elif financial_statement == "Balance Sheet":
                financial_data = yf.Ticker(stock_symbol).quarterly_balance_sheet
            elif financial_statement == "Cash Flow":
                financial_data = yf.Ticker(stock_symbol).quarterly_cashflow

        # Display Financial Data
        st.write(financial_data)

    # Monte Carlo Simulation Tab
    with tabs[3]:
        st.subheader(f"🎲 {stock_symbol} Monte Carlo Simulation")
        st.markdown("---")

        # Simulation Parameters
        num_simulations_options = [200, 500, 1000]
        num_simulations = st.selectbox("Select Number of Simulations:", num_simulations_options)

        time_horizon_options = [30, 60, 90]
        time_horizon = st.selectbox("Select Time Horizon (days):", time_horizon_options)

        # Monte Carlo Simulation
        @st.cache_data(ttl=3600)
        def monte_carlo_simulation(data, num_simulations, time_horizon, confidence_level=0.95):
            returns = data["Close"].pct_change().dropna()
            last_price = data["Close"][-1]
            mean_return = returns.mean()
            std_return = returns.std()

            simulation_df = pd.DataFrame()

            for _ in range(num_simulations):
                price_series = [last_price]

                for _ in range(time_horizon):
                    price_series.append(price_series[-1] * (1 + np.random.normal(mean_return, std_return)))

                simulation_df[f"Sim {_}"] = price_series

            final_prices = simulation_df.iloc[-1]
            var_95 = np.percentile(final_prices, 100 * (1 - confidence_level))

            return simulation_df, var_95

        if st.button("Run Simulation"):
            simulation_results, var_95 = monte_carlo_simulation(stock_data, num_simulations, time_horizon)

            # Plot Simulation Results
            st.subheader("Simulation Results")
            fig = go.Figure()
            for col in simulation_results.columns:
                fig.add_trace(go.Scatter(x=simulation_results.index, y=simulation_results[col], mode="lines", opacity=0.2, name=col))

            fig.update_layout(
                title=f"{stock_symbol} Monte Carlo Simulation for {time_horizon} days",
                xaxis=dict(title="Days"),
                yaxis=dict(title="Price"),
                legend=dict(x=1, y=1),
                hovermode="x unified",
                template="plotly_white",
            )

            st.plotly_chart(fig)

            # Calculate Value at Risk (VaR)
            st.subheader("Value at Risk (VaR) at 95% Confidence Interval")
            st.write(f"VaR at 95% confidence interval: ${var_95:.2f}")

    # Analytics Tab
    with tabs[4]:
        st.subheader(f"📉 {stock_symbol} Analytics")
        st.markdown("---")

        stock_data["Daily Return"] = stock_data["Close"].pct_change()
        st.write(stock_data["Daily Return"].dropna())

        # Add more analytics
        st.subheader("Rolling Statistics")
        stock_data["Rolling Mean"] = stock_data["Close"].rolling(window=30).mean()
        stock_data["Rolling Std"] = stock_data["Close"].rolling(window=30).std()
        st.write(stock_data[["Rolling Mean", "Rolling Std"]])

        # Plot rolling statistics
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Rolling Mean"], mode="lines", name="Rolling Mean"))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Rolling Std"], mode="lines", name="Rolling Std"))
        st.plotly_chart(fig)

else:
    st.warning("Please select a valid stock symbol.")
