
import yfinance as yf

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as py

def yahoo_download(tickers, start, end):
    data_out_long = []
    data = yf.download(tickers, start=start, end=end)
    data_out = data.stack(level=0).reset_index()
    data_out_long = data_out.melt(id_vars=['Date', 'level_1'], value_vars=tickers, var_name='Symbol')
    return data_out_long

fig, ax = plt.subplots()

#The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions.
# Get the data of the stock AAPL
# data = yf.download('AAPL', '2016-01-01', '2018-01-01')

reference_table = pd.read_csv(r'C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\constituents_csv.csv')
sectors = pd.DataFrame(reference_table["Sector"].unique())
sectors = sectors.rename(columns={0:"Sector"})
sectors.sort_values(by='Sector')

#tickers = list(reference_table["Symbol"])

tickers = ["AAPL", "GOOG", "MSFT"]
benchmark = "SPY"
start = "2017-01-01"
end = "2017-04-30"

data_stocks = yahoo_download(tickers, start, end)
data_benchmark = yahoo_download(benchmark, start, end)

data_stocks.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooStockData.csv")
data_benchmark.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooBenchmarkData.csv")

sectors_interest = 'Information Technology'
sectors_ticker = pd.DataFrame(reference_table[["Sector","Symbol"]])
#data = yf.download(tickers, start="2017-01-01", end="2017-04-30")
#data = yf.download(tickers, start="2017-01-01", end="2017-04-30")
#Should save the data to a csv file to prevent having to keep downloading from Yahoo
#data_out = data.stack(level=0).reset_index()
#data_out.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooData_before.csv")
#data_out_long = data_out.melt(id_vars=['Date', 'level_1'], value_vars=tickers, var_name='Symbol')
#data_out_long = data_out.melt(id_vars=['Date', 'level_1'], value_vars=tickers, var_name = 'Symbol'
#data_out_long.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooData.csv")

#data_out_long = pd.read_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooStockData.csv")

merged_data = data_stocks.merge(sectors_ticker, on='Symbol')

interest_only_close = merged_data.loc[(merged_data['level_1']=="Close") & (merged_data['Sector']==sectors_interest)]

benchmark_close = data_benchmark.loc[data_benchmark['level_1']=="Close"]
first_benchmark = benchmark_close.value.iloc[0]
agg_benchmark = benchmark_close.groupby('Date').sum()

agg_interest = interest_only_close.groupby('Date').sum()
first_price = agg_interest.value.iloc[0]
agg_interest_index = agg_interest.value.div(first_price).mul(100)

benchmark_index = agg_benchmark.div(first_benchmark).mul(100)

ax.plot(agg_interest_index, color = 'red', label="Selected")
ax.plot(benchmark_index, color = 'blue', label = "Benchmark")
ax.legend()
plt.show()

# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()
