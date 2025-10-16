#============================================================================================
# Stock data retrieval (from yahoo finance)
#============================================================================================
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stockName = "^SPX"
#stockName = "^DJI"

def get_stock_data(period="6y", interval="1d"):
    try:
        stock = yf.Ticker(stockName)
        hist = stock.history(period=period, interval=interval)
        hist = hist.reset_index()
        return hist
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def split_data_by_date(data, recent_years=1):
    """
    Splits the dataframe into two: the most recent `recent_years` of data,
    and the remaining older data.
    """
    if data is None or data.empty:
        return None, None

    cutoff_date = data["Date"].max() - pd.DateOffset(years=recent_years)

    recent_data = data[data["Date"] > cutoff_date].copy()
    older_data = data[data["Date"] <= cutoff_date].copy()

    return recent_data, older_data

def plot_stock_data(data, title="Stock Data"):
    if data is not None and not data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Close'], linewidth=2)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Fetching 6 years of data...")
    full_data = get_stock_data(period="6y", interval="1wk")

    if full_data is not None and not full_data.empty:
        print(f"Total records fetched: {len(full_data)}")
        print(f"Date range: {full_data['Date'].min().date()} to {full_data['Date'].max().date()}")

        last_year_data, previous_five_years_data = split_data_by_date(full_data, recent_years=1)

        if last_year_data is not None:
            print(f"\nMost recent year: {len(last_year_data)} records")
            last_year_data.to_csv(f"Stock Time Series/Data/{stockName}_last_year.csv", index=False)
            print("Saved: djia_last_year.csv")
            plot_stock_data(last_year_data, title="DJIA – Last Year")

        if previous_five_years_data is not None:
            print(f"\nPrevious 5 years: {len(previous_five_years_data)} records")
            previous_five_years_data.to_csv(f"Stock Time Series/Data/{stockName}_prior_five_years.csv", index=False)
            print("Saved: djia_prior_five_years.csv")
            plot_stock_data(previous_five_years_data, title="DJIA – Prior 5 Years")

    else:
        print("Failed to fetch stock data.")

