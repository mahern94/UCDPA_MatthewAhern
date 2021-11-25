import yfinance as yf
import numpy as np
<<<<<<< HEAD
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
=======
>>>>>>> origin/master
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl as py

#Custom function to simplify the downloading process from yfinance online API
def yahoo_download(tickers, start, end):
    data_out_long = []
    #This is standard coding to retrieve data from yfinance online API
    data = yf.download(tickers, start=start, end=end)
    data_out = data.stack(level=0).reset_index()
    data_out_long = data_out.melt(id_vars=['Date', 'level_1'], value_vars=tickers, var_name='Symbol')
    return data_out_long

#The adjusted closing price amends a stock's closing price to reflect that stock's value after accounting for any corporate actions.


#Import csv file 'constituents_csv.csv' which contains the ticker symbols
reference_table = pd.read_csv(r'C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\constituents_csv.csv')
sectors = pd.DataFrame(reference_table["Sector"].unique())
sectors = sectors.rename(columns={0:"Sector"})
sectors.sort_values(by='Sector')

<<<<<<< HEAD

#Save tickers as a list of individual stock ticker symbols to aid easy iteration when using 'for' loops
tickers = ["NXPI", "MMM", "CAH","AMZN","LUMN","UDR","COP","CNP","HAS","OKE","VMC","EBAY","DLR","CB","BA","GPC","WRK","TEL","RL","TFC"]

tickers_df = pd.DataFrame(tickers, columns=['Symbol'])
#Select the S&P500 index as our benchmark
benchmark = 'SPY'
start = "2011-01-01"
=======
#tickers = list(reference_table["Symbol"])

#tickers = ["AAPL", "GOOG", "MSFT", "ATVI"]
#tickers = ["ATVI", columns = "Ticker"]
tickers = ["NXPI", "MMM", "CAH","AMZN","LUMN","UDR","COP","CNP","HAS","OKE","VMC","EBAY","DLR","CB","BA","GPC","WRK","TEL","RL","TFC"]

sectors_selected = ["Consumer Discretionary", "Energy"]
tickers_df = pd.DataFrame(tickers, columns=['Symbol'])
benchmark = 'SPY'
start = "2015-01-01"
>>>>>>> origin/master
end = "2020-12-31"

data_stocks = yahoo_download(tickers, start, end)
data_benchmark = yahoo_download(benchmark, start, end)
data_benchmark.set_index('Date')

data_stocks.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooStockData.csv")
data_benchmark.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooBenchmarkData.csv")
<<<<<<< HEAD

if len(tickers_df) == 1:
    data_stocks['Symbol'] = tickers

=======
if len(tickers_df) == 1:
    data_stocks['Symbol'] = tickers



>>>>>>> origin/master
sectors_ticker = pd.DataFrame(reference_table[["Sector","Symbol"]])
selected_ticker_sector = tickers_df.merge(sectors_ticker, on='Symbol')
merged_data = data_stocks.merge(sectors_ticker, on='Symbol')
sectors_selected = merged_data['Sector'].unique()

#Drop duplicates & sort by Ticker Symbol
stock_sector_info = merged_data[["Symbol","Sector"]].drop_duplicates(subset=["Symbol","Sector"]).reset_index(drop = True).sort_values("Symbol")

<<<<<<< HEAD
#FIND THE NUMBER OF STOCKS IN EACH SECTOR, THEN USE ITERROWS TO CREATE A PRINT STATEMENT WHICH ITERATES OVER EACH SECTOR & GIVES THE NUMBER OF STOCKS PER SECTOR
#AS PER EXAMPLE HERE: https://pythonexamples.org/pandas-dataframe-iterate-rows-iterrows/

#Give the composition of the portfolio:
def my_fmt(x):
    print(x)
    return '{:.3}%\n'.format(x, total*x/100)
portfolio_sector_composition = stock_sector_info.value_counts("Sector")
total = portfolio_sector_composition.sum()
portfolio_sector_composition.plot(kind="pie", title = f"Proportion of Stocks by Sector for %d stocks in the portfolio" % total, autopct = my_fmt, ylabel = '')
plt.show()

#Get closing prices of the benchmark & normalize
=======
#Give the composition of the portfolio:
portfolio_sector_composition = stock_sector_info.value_counts("Sector")
portfolio_sector_composition.plot(kind="pie")
plt.show()

sectors_interest = []

#benchmark_stats
>>>>>>> origin/master
benchmark_close = data_benchmark.loc[data_benchmark['level_1']=="Close"]
benchmark_close = benchmark_close.set_index('Date')
first_benchmark = benchmark_close.value.iloc[0]
agg_benchmark = benchmark_close.groupby('Date').sum()
benchmark_index_normalized = agg_benchmark.div(first_benchmark).mul(100)
benchmark_index = agg_benchmark

<<<<<<< HEAD
benchmark_close['pct_change'] = benchmark_close.value.pct_change().mul(100)
benchmark_close['mean30D'] = benchmark_close['value'].rolling(window='30D').mean()
benchmark_close_stats = benchmark_close['value'].rolling(window='30D').agg(['mean','std'])
benchmark_close_stats.plot(subplots = True, title = "30-Day Rolling Mean & Standard Deviation for S&P500 Benchmark", ylabel="Return")


benchmark_rolling = benchmark_close['value'].rolling(window='360D')
q10 = benchmark_rolling.quantile(0.1).to_frame('q10')
median = benchmark_rolling.median().to_frame('median')
q90 = benchmark_rolling.quantile(0.9).to_frame('q90')
pd.concat([q10, median, q90], axis=1).plot(title = "360-Day rolling quantiles of the S&P 500 benchmark")


benchmark_summ_stats = benchmark_close['pct_change'].dropna().describe()

fig, ax = plt.subplots()
stocks_close = merged_data.loc[merged_data['level_1']=="Close"]
stocks_close = stocks_close.reset_index()
stocks_close_annual_returns = pd.DataFrame()
stocks_close_pivot = stocks_close.pivot_table(values="value", index="Date", columns="Symbol", fill_value = 0)
stocks_close_annual_returns = stocks_close_pivot.pct_change(periods=360).mul(100)
stocks_close_annual_returns.replace([np.inf, -np.inf], np.nan, inplace = True)
stocks_close_std = stocks_close_annual_returns.std().sort_values(ascending=True)

#Plot the standard deviations as a bar chart
ax.bar(stocks_close_std.index,stocks_close_std)
ax.set_xticklabels(stocks_close_std.index, rotation = 90)
ax.set_ylabel("Standard Deviation")
ax.set_title("Standard Deviation by Stock")
plt.show()


#Show the annualized standard deviations for each year for Boeing to check which years were most volatile
stocks_close_boeing = stocks_close_annual_returns["BA"].dropna()
stocks_close_boeing = stocks_close_boeing.reset_index()
stocks_close_boeing = pd.DataFrame(stocks_close_boeing)
stocks_close_boeing['Year'] = pd.DatetimeIndex(stocks_close_boeing['Date']).year
#Use the .agg method & numpy to create the aggregate statistics
stocks_close_boeing_info = stocks_close_boeing.groupby("Year")["BA"].agg([np.mean, np.std])

fig1, ax1 = plt.subplots(2,1)
width = 0.3
N = stocks_close_boeing['Year'].drop_duplicates().value_counts().sum()
ind = np.arange(N)
stocks_close_boeing = stocks_close_boeing.set_index('Date')
stocks_close_boeing = stocks_close_boeing.drop('Year', 1)
ax1[0].plot(stocks_close_boeing)
ax1[0].set_ylabel("Boeing (""BA"") annualized stock returns")
ax1[0].set_title("Boeing (""BA"") Annualized Stock Returns, Mean and Standard Deviation by Year")
plt.show()
ax1[1].bar(ind,stocks_close_boeing_info["mean"], width, label = "Annualized Mean Return")
ax1[1].bar(ind + width,stocks_close_boeing_info["std"], width, label = "Annualized Standard Deviation")
#Set the xticks as a list to aid the labelling of the tick labels
xticks = list(stocks_close_boeing_info.index)
ax1[1].set_xticks(ind + width / 2)
ax1[1].set_xticklabels(xticks)
ax1[1].set_ylabel("Mean / Standard Deviation")
ax1[1].legend(loc='best')
plt.show()

stocks_by_sector_normalized = pd.DataFrame()
stocks_by_sector = pd.DataFrame()
for sector in sectors_selected:
    sector_iter = stocks_close.loc[stocks_close['Sector']==sector].reset_index(drop=True)
    sector_iter = sector_iter.groupby('Date').sum()
    sector_iter_value = pd.DataFrame(sector_iter.value)
    first_price = sector_iter.value.iloc[0]
    sector_iter_index = pd.DataFrame(sector_iter_value.value.div(first_price).mul(100))
    sector_iter_index.rename(columns={'value': sector}, inplace=True)
    stocks_by_sector = pd.concat([stocks_by_sector, sector_iter_index.mul(first_price).div(100)], axis = 1)
    stocks_by_sector_normalized = pd.concat([stocks_by_sector_normalized, sector_iter_index], axis=1)

benchmark_index_normalized.rename(columns={'value': 'Benchmark'}, inplace=True)
benchmark_index.rename(columns={'value': 'Benchmark'}, inplace=True)
stocks_by_sector_normalized = pd.concat([stocks_by_sector_normalized, benchmark_index_normalized], axis=1)
stocks_by_sector = pd.concat([stocks_by_sector, benchmark_index], axis=1)
#Calculate the summary statistics of the stocks_by_sector_normalized & the benchmark to compare
#Set the columns as an array which we can loop through the column names
columns_sel = stocks_by_sector_normalized.columns.to_numpy()
stocks_by_sector_normalized_pct_change = pd.DataFrame()
stocks_by_sector_pct_change = pd.DataFrame()

for label in columns_sel:
    stocks_by_sector_normalized_pct_change[label] = stocks_by_sector_normalized[label].pct_change(periods=360).mul(100)
    #Get the 360-day percent change for the sector-level stock price returns
    stocks_by_sector_pct_change[label] = stocks_by_sector[label].pct_change(periods=360).mul(100)

stocks_by_sector_normalized_pct_change = stocks_by_sector_normalized_pct_change.dropna()

#Create a correlation matrix which examines the correlations between the sectors underlying the portfolio
stocks_by_sector_pct_change = stocks_by_sector_pct_change.dropna()
correlations = stocks_by_sector_pct_change.corr()
sns.heatmap(correlations, annot = True)
plt.xticks(rotation=90)
plt.title('Return Correlations')
plt.show()

#Get summary statistics of the annualized returns of the selected portfolio & the benchmark
print(stocks_by_sector_normalized_pct_change.describe())
#Create boxplots of the sector-level annualized returns to visualize the...
fig3, ax3 = plt.subplots()
ax3.boxplot(stocks_by_sector_normalized_pct_change)
ax3.set_xticklabels(columns_sel, rotation=90)

#Defining a user-defined function to calculate the running return & multi-period return
def multi_period_return(period_returns):
    return np.prod(period_returns+1) - 1

#Comparing the normalized portfolio returns (grouped by sector) with the S&P 500 benchmark
fig4,ax4 = plt.subplots(2,1)
stocks_by_sector_normalized_pct_change.apply(multi_period_return)
stocks_by_sector_normalized = stocks_by_sector_normalized.drop('Benchmark', 1)
stocks_by_sector_normalized_labels = stocks_by_sector_normalized.columns.to_numpy()
ax4[0].plot(stocks_by_sector_normalized, label = stocks_by_sector_normalized_labels)
ax4[0].plot(benchmark_index_normalized, label = 'Benchmark', color = 'Red')
ax4[0].legend(loc='upper left')

agg_stocks = stocks_close.groupby('Date').sum()
first_price = agg_stocks.value.iloc[0]
agg_stocks_normalized = agg_stocks.value.div(first_price).mul(100)
ax4[1].plot(agg_stocks_normalized, color = 'blue', label="Selected Portfolio")
ax4[1].plot(benchmark_index_normalized, color = 'red', label = "Benchmark")
ax4[1].legend(loc='upper left')
plt.show()

=======
#Need to calculate the percent change on the benchmark
#benchmark_close['pct_change'] = benchmark_close.value.pct_change().mul(100)
#benchmark_close['mean30D'] = benchmark_close['value'].rolling(window='30D').mean()
#benchmark_close['mean30'] = benchmark_close['value'].rolling(window=30).mean()
#benchmark_close_stats = benchmark_close['value'].rolling(window='30D').agg(['mean','std'])
#benchmark_close_stats.plot(subplots = True)

#At what points is the benchmark at it's most volatile?
#benchmark_rolling = benchmark_close['value'].rolling(window='360D')
#q10 = benchmark_rolling.quantile(0.1).to_frame('q10')
#median = benchmark_rolling.median().to_frame('median')
#q90 = benchmark_rolling.quantile(0.9).to_frame('q90')
#pd.concat([q10, median, q90], axis=1).plot()


#benchmark_close.to_csv(r"C:\Users\matth\OneDrive\Documents\Data Analytics for Finance\Project\YahooBenchmarkData_rollingavg.csv")
#benchmark_summ_stats = benchmark_close['pct_change'].dropna().describe()

#Need to investigate the 30-day rolling average return on the stocks...is there potential to investigate vs inflation?


if sectors_interest != []:
    interest_only_close = merged_data.loc[(merged_data['level_1']=="Close") & (merged_data['Sector']==sectors_interest)]
    agg_interest = interest_only_close.groupby('Date').sum()
    first_price = agg_interest.value.iloc[0]
    agg_interest_index = agg_interest.value.div(first_price).mul(100)
    #fig, ax = plt.subplots()
   # ax.plot(agg_interest_index, color = 'red', label="Selected")
   # ax.plot(benchmark_index, color = 'blue', label = "Benchmark")
  #  ax.legend()
   # plt.show()

else:
    #fig, ax = plt.subplots(2,1)
    #COULD CREATE DICTIONARY THAT HAS EACH SECTOR??
    stocks_close = merged_data.loc[merged_data['level_1']=="Close"]
    stocks_close = stocks_close.reset_index()
    #stocks_close = stocks_close.set_index('Date')
    stocks_close_annual_returns = pd.DataFrame()
    stocks_close_pivot = stocks_close.pivot_table(values="value", index="Date", columns="Symbol", fill_value = 0)
    stocks_close_annual_returns = stocks_close_pivot.pct_change(periods=360).mul(100)
    stocks_close_annual_returns.replace([np.inf, -np.inf], np.nan, inplace = True)
    stocks_close_std = stocks_close_annual_returns.std().sort_values(ascending=True)
    stocks_close_std.plot(kind='bar')
    plt.show()

    #Show the annualized standard deviations for each year for Boeing to check how
    stocks_close_boeing = stocks_close_annual_returns["BA"].dropna()
    stocks_close_boeing = stocks_close_boeing.reset_index()
    stocks_close_boeing = pd.DataFrame(stocks_close_boeing)
    stocks_close_boeing['Year'] = pd.DatetimeIndex(stocks_close_boeing['Date']).year
    stocks_close_boeing_info = stocks_close_boeing.groupby("Year")["BA"].agg([np.mean, np.std])
    stocks_close_boeing_info.plot(kind='bar')
    plt.show()



    #THEN CAN SHOW A CHART SHOWING THE ANNUALIZED STANDARD DEVIATIONS TO CHECK WHICH YEAR CONTRIBUTED MOST TO THE VOLATILITY

    #LOOK AT THE TRADING VOLUMES ALSO - THEN CLOSE OUT THE PROJECT...NO NEED FOR FURTHER WORK


    #WHICH SECTORS/STOCKS PERFORMED WELL VS THE INDEX AND WHICH OTHERS DIDN'T DO SO WELL?
    #WHAT IS THE MOST VOLATILE STOCK IN THE PORTFOLIO?
    #WHICH IS THE MOST VOLATILE SECTOR? WHY WOULD THIS BE THE CASE?
    #WHICH STOCKS PRODUCE THE HIGHEST RETURNS?
    #WHICH STOCKS WOULD HAVE OUT-PERFORMED INFLATION?
    #HOW IS THE PORTFOLIO COMPOSED?
    #WHAT DOES THE BOX-PLOTS AS PER BELOW TELL US ABOUT THE SECTOR-WIDE ANALYSIS?
    #WHICH STOCK HAS THE HIGHEST AVERAGE TRADING VOLUME?
    #WHAT WAS THE HIGHEST VOLUME TRADED IN A GIVEN DAY?
    #IS THERE ANY EVIDENCE OF SEASONALITY IN THE VOLUMES TRADED OVER GIVEN QUARTERS?

    #REMAINING PARTS OF MARKING SCHEME NOT ADDRESSED IN THIS CODE: ITERROWS, DROP DUPLICATES (GENERATE A LIST OF DATETIMES &MERGE THEN DROP??, DICTIONARIES

    #Possible to create a dictionary that shows the mean price per year or something???

    stocks_by_sector_normalized = pd.DataFrame()
    stocks_by_sector = pd.DataFrame()
    for sector in sectors_selected:
        sector_iter = stocks_close.loc[stocks_close['Sector']==sector].reset_index(drop=True)
        sector_iter = sector_iter.groupby('Date').sum()
        sector_iter_value = pd.DataFrame(sector_iter.value)
        first_price = sector_iter.value.iloc[0]
        sector_iter_index = pd.DataFrame(sector_iter_value.value.div(first_price).mul(100))
        sector_iter_index.rename(columns={'value': sector}, inplace=True)
        stocks_by_sector = pd.concat([stocks_by_sector, sector_iter_index.mul(first_price).div(100)], axis = 1)
        stocks_by_sector_normalized = pd.concat([stocks_by_sector_normalized, sector_iter_index], axis=1)

    benchmark_index_normalized.rename(columns={'value': 'Benchmark'}, inplace=True)
    benchmark_index.rename(columns={'value': 'Benchmark'}, inplace=True)
    stocks_by_sector_normalized = pd.concat([stocks_by_sector_normalized, benchmark_index_normalized], axis=1)
    stocks_by_sector = pd.concat([stocks_by_sector, benchmark_index], axis=1)
    #Calculate the summary statistics of the stocks_by_sector_normalized & the benchmark to compare
    #Set the columns as an array which we can loop through the column names
    columns_sel = stocks_by_sector_normalized.columns.to_numpy()
    stocks_by_sector_normalized_pct_change = pd.DataFrame()
    stocks_by_sector_pct_change = pd.DataFrame()

    for label in columns_sel:
        stocks_by_sector_normalized_pct_change[label] = stocks_by_sector_normalized[label].pct_change(periods=360).mul(100)
        stocks_by_sector_pct_change[label] = stocks_by_sector[label].pct_change(periods=360).mul(100)
    stocks_by_sector_normalized_pct_change = stocks_by_sector_normalized_pct_change.dropna()
    stocks_by_sector_pct_change = stocks_by_sector_pct_change.dropna()
    correlations = stocks_by_sector_pct_change.corr()

    sns.heatmap(correlations, annot = True)
    plt.xticks(rotation=45)
    plt.title('Return Correlations')
    plt.show()


    #Get summary statistics of the selected portfolio & the benchmark
    stocks_by_sector_pct_change.describe()
   # stocks_by_sector_normalized_pct_change_chart = stocks_by_sector_normalized_pct_change
    fig1, ax1 = plt.subplots()
    ax1.boxplot(stocks_by_sector_pct_change)
    ax1.set_xticklabels(columns_sel, rotation=45)


    #Get the correlations of the selected portfolio with the benchmark...



    #Which stocks contribute most to the movement?? -> may need to look at market capitalizations....

#Calculating the running return & multi-period return
    def multi_period_return(period_returns):
        return np.prod(period_returns+1) - 1


    stocks_by_sector_normalized_pct_change.apply(multi_period_return)
    stocks_by_sector_normalized_labels = stocks_by_sector_normalized.columns.to_numpy()
    ax[0].plot(stocks_by_sector_normalized, label = stocks_by_sector_normalized_labels)
    ax[0].plot(benchmark_index_normalized, label = 'Benchmark', color = 'Red')
    ax[0].legend()
    plt.show()

    agg_interest = stocks_close.groupby('Date').sum()
    first_price = agg_interest.value.iloc[0]
    agg_interest_index = agg_interest.value.div(first_price).mul(100)
    ax[1].plot(agg_interest_index, color = 'red', label="Selected")
    ax[1].plot(benchmark_index_normalized, color = 'blue', label = "Benchmark")
    ax[1].legend()
    plt.show()

# Plot the close price of the AAPL
#data.Close.plot()
#plt.show()
>>>>>>> origin/master
