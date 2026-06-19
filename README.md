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

<p align="center">
  <img src="outputs/methodology.png" width="40%">
</p>

## 1. Data Scraping:
   The financial data was extracted using the Python yfinance library, the data was scraped in a dictionary form containg the ticker as keys and the name of the stock as values. The raw data fetched consisted of OHLCV (Open High Low Close Volume) values indexed by its date in the format datetime64[ns]. After the raw data scraping, it was then handled and appended into 4 separated lists containing the useful values for the model: symbols, names, close prices and open prices. The lists containing the close and open prices where then manipulated and stacked in order to proceed for the featuring engineering step.
   
## 2. Feature Engineering: 
The features used in the model are the daily price variations of the 54 stocks. In order to obtain these values, the difference between the close price and the open price is defined as variations for each trading day:
>variation = close_prices - open_prices

This measure was used to capture the intraday directional movements of each stock. Positive values indicates that the stock gained value during that session, while negative values indicate a loss. Thereafter, these variation values were stored in a pandas DataFrame and transposed in order to be cleaned and handled.

## 3. Data Cleaning: 
In order to produce a complete matrix of shape n_stocks x n_days without NaN values, the missing observations from non-trading days were handled through forward-fill followed by a backward-fill interpolation.

## 3. Standardisation:
The z-score scaling was applied to the dataset used in the Graphical Lasso model to prevent the model from being influenced by raw value inflation caused by differing means, and to ensure equal weighting across stocks for the regularisation penalty.

## 4. Graphical Lasso CV:
After cleaning and applying z-score scaling to the dataset, it was then fitted into the Graphical Lasso CV method to estimate the sparse precision matrix (inverse covariance matrix) for Gaussian Graphical models, with a cross-validated choice of the L1 penalty from a logarithmically spaced range of candidate values (alphas). The Python library scikit-learn was used to implement this method.

## 5. Affinity propagation:
The final exploratory data analysis was performed using the Afinitty Propagation clustering algorithm, which automatically identifies clusters and their exemplars (representative points) without requiring the number of clusters to be specified in advance.

## 6. Locally Linear Embedding:
Locally Linear Embedding (LLE) was applied to the data to reduce it to a 2D space while preserving its essential structure. LLE dimensionality reduction was used to facilitate cluster visualisation in a 2D map coloured by cluster, by producing coordinates for each stock as a node.

## 7. Cluster Visualisation:
A 2D stock map containing coloured stock nodes and their connections in a network form was produced to facilitate visualisation in a simple and intuitive way. The partial correlations used to form the network were derived from the precision matrix to determine the strength of pairwise relationships between stocks, with absolute partial correlations greater than 0.02 being retained. The node coordinates from LLE and the relationships from partial correlations were then plotted using the matplotlib library. The node size is scaled according to the inverse-variance weighting of each stock. The node colour is assigned according to cluster membership. Edge thickness and colour intensity are scaled according to the strength of the partial correlations, allowing both cluster structure and the relative strength of pairwise relationships to be interpreted visually.


 
