# Saudi Stock Market Clustering - Tadawul (2022-2025)
> Unsupervised ML applied to the Saudi Exchange (Tadawul) to uncover hidden correlation structures among 54 stocks across 8 sectors. 
>
 ## Executive Summary
 In this algorithm both Graphical Lasso CV and Affinity Propagation methodology were applied to cluster the present Saudi Stocks.The features are their daily variations from a time-series dataset throughout three years, covering data from 54 stocks across 8 sectors: Energy, Banking, Telecommunications, Retail, Real State, Healthcare, Cement and Transportation.
 The main goal of this project is to identify clusters of stocks that move together in the Tadawul, not based on their sectors, but rather purely based on their price behaviour, and to use the results from this research for a smarter diversification analysis and portfolio optimisation. The results are presented in a graphical form in order to facilitate the understanding through data visualisation.

## Why not Markowitz Mean-Variance Portfolio Theory?
 Markowitz Mean-variance Portfolio Theory is the classical approach to portfolio construction. However, according to Seregina & Lee (2023), when the number of stocks (p) in the dataset is larger than the number of observations (n), obtaining portfolio weights leads to unstable investment allocations, known as the Markowitz' curse.
 Alternatively, Graphical Lasso is a powerful tool to estimate a high-dimensional inverse covariance matrix in order to provide consistent and stable estimations fo asset allocations, based on Friedman et al (2008) and Seregina & Lee (2023). Graphical Lasso solves the Markowitz' curse by estimating the sparse inverse covariance matrix directly, penalizing weak correlations. Goto & Xu (2015) also obtained positive results aand avhieved significant out-of-sample risk reduction and higher returns after applying Graphical Lasso.

## Methodology 
> The present algorithm consists mainly of seven steps. Subsequently, these steps are described:

## 1. Data Scraping:
   The financial data was extracted using the Python yfinance library, the data was scraped in a dictionary form containg the ticker as keys and the name of the stock as values. The raw data fetched consisted of OHLCV (Open High Low Close Volume) values indexed by its date in the format datetime64[ns]. After the raw data scraping, it was then handled and appended into 4 separated lists containing the useful values for the model: symbols, names, close prices and open prices. The lists containing the close and open prices where then manipulated and stacked in order to proceed for the featuring engineering step.
   
## 2. Feature Engineering: 
The features used in the model are the daily price variations of the 54 stocks. In order to obtain these values, the difference between the close price and the open price is defined as variations for each trading day:
>variation = close_prices - open_prices

This measure was used to capture the intraday directional movements of each stock. Positive values indicates that the stock gained value during that session, while negative values indicate a loss. Thereafter, these variation values were stored in a pandas DataFrame and transposed in order to be cleaned and handled.

## 3. Data Cleaning: 

 
