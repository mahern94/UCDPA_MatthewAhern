
import yfinance as yf

import pandas as pd
import matplotlib.pyplot as plt


#The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions.
# Get the data of the stock AAPL
# data = yf.download('AAPL', '2016-01-01', '2018-01-01')

reference_table = pd.read_csv(r'C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\constituents_csv.csv')
tickers = list(reference_table["Symbol"])

sectors = pd.DataFrame(reference_table["Sector"].unique())
sectors = sectors.rename(columns={0:"Sector"})
sectors.sort_values(by='Sector')

print(sectors.info())
#tickers = ['MSFT', 'AAPL', 'GOOG']

sectors_interest = ['Information Technology']

sectors_ticker = pd.DataFrame(reference_table[["Sector","Symbol"]])



data = yf.download(tickers, start="2017-01-01", end="2017-04-30")
data_out = data.stack(level=0).reset_index()
data_out_long = data_out.melt(id_vars=['Date', 'level_1'], value_vars=tickers, var_name = 'Symbol')

merged_data = data_out_long.merge(sectors_ticker, on='Symbol')

merged_data.loc[(merged_data['level_1']=="Close")&(merged_data['Sector']=="Industrials")]

agg_industrial = industrial_only_close.groupby('Date').sum()

agg_industrial.plot()

# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()
