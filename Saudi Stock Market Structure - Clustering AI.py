# -*- coding: utf-8 -*-
"""
Created on Sun May 31 02:56:48 2026

@author: Ericka
"""
#This algorithm applies several unsupervised learning techniques to extract
# the saudi stock market strucuture from variations in historical quotes

import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import affinity_propagation
from sklearn.manifold import LocallyLinearEmbedding

symbol_dict = {
    #ENERGY
    "2222.SR": "Saudi Aramco",
    "1211.SR": "Saudi Arabian Mining",
    "2380.SR": "Rabigh Refining",
    "2290.SR": "Yanbu Petrochemical",
    "2223.SR": "Saudi Aramco Base Oil - Luberef",
    "1322.SR": "AMAK Mining",
    #BANKING
    "1120.SR": "Al Rajhi Bank",
    "1180.SR": "Saudi National Bank",
    "1010.SR": "Riyad Bank",
    "1150.SR": "Alinma Bank",
    "1060.SR": "Saudi Awwal Bank",
    "1050.SR": "Banque Saudi Fransi",
    "1080.SR": "Arab National Bank",
    "1140.SR": "Bank Al Bilad",
    "1030.SR": "Saudi Investment Bank",
    "1020.SR": "Bank Al Jazira",
    #TELECOM
    "7010.SR": "STC",
    "7020.SR": "MOBILY",
    "7030.SR": "ZAIN",
    "7040.SR": "GO",
    #RETAIL
    "4190.SR": "JARIR",
    "4161.SR": "BinDawood",
    "4001.SR": "Al Othaim Markets",
    "4050.SR": "Saudi Automative Services Co.",
    "4165.SR": "Al Majed Oud",
    #REAL STATE
    "4300.SR": "Dar Al Arkan",
    "4250.SR": "Jabal Omar",
    "4020.SR": "Saudi Real State Co.",
    "4100.SR": "Makkah Construction",
    #HEALTHCARE
    "4013.SR": "Dr. Suleiman Al Habib",
    "4002.SR": "Mouwasat",
    "4015.SR": "Jamjoom Pharmaceuticals",
    "1212.SR": "Astra",
    "4004.SR": "Dallah Healthcare",
    "4017.SR": "Dr. Soliman Abdul Kader",
    #CEMENT
    "3040.SR": "Qassim Cement Co.",
    "3020.SR": "Yamamah Cement Co.",
    "3030.SR": "Saudi Cement Co.",
    "3050.SR": "Southern Province Cement Co.",
    "3092.SR": "Riyadh Cement Co.",
    "3060.SR": "Yanbu Cement Co.",
    "3010.SR": "Arabian Cement Co.",
    "3080.SR": "Eastern Province Cement Co.",
    "3003.SR": "City Cement Co.",
    "3004.SR": "Northern Region Cement Co.",
    "3002.SR": "Najran Cement Co.",
    "3005.SR": "Umm Al-Qura Cement Co.",
    "3090.SR": "Tabuk Cement Co.",
    "3091.SR": "Al Jouf Cement Co.",
    #TRANSPORTATION
    "4030.SR": "National Shipping Co.",
    "4263.SR": "SAL Logistics",
    "4264.SR": "Flynas",
    "4031.SR": "Saudi Ground Services",
    "4040.SR": "Saudi Public Transport"
    }

symbols = np.array(list(symbol_dict.keys()))
names = np.array(list(symbol_dict.values()))

START = "2022-01-01"
END = "2025-12-31"

valid_names = []
valid_symbols = []
close_list = []
open_list = []

print(f'Fetching {len(symbols)} Saudi stocks from yfinance...', file=sys.stderr)

for sym,name in zip(symbols, names):
    try:
        df = yf.download(sym, start=START, end=END, progress = False, auto_adjust=True)
        if df.empty or len(df) < 100:
            print(f'SKIP {sym}: insufficient data ({len(df)} rows)', file=sys.stderr)
            continue
        valid_symbols.append(sym)
        valid_names.append(name)
        close_list.append(df['Close'].values.flatten())
        open_list.append(df['Open'].values.flatten())
        print(f"  OK   {sym} ({name})  — {len(df)} rows", file=sys.stderr)
    except Exception as exc:
        print(f"  ERR  {sym}: {exc}", file=sys.stderr)  
        
min_len = min(len(c) for c in close_list)
close_prices = np.vstack([c[:min_len] for c in close_list])
open_prices = np.vstack([o[:min_len] for o in open_list])

valid_symbols = np.array(valid_symbols)
valid_names   = np.array(valid_names)
 
print(f"\nSuccessfully loaded {len(valid_symbols)} stocks, {min_len} trading days.", file=sys.stderr)

variation = close_prices - open_prices
variation_df = pd.DataFrame(variation.T)
variation_df = variation_df.ffill().bfill()
variation = variation_df.values.T

assert variation.ndim == 2, f"Expected 2D array, got shape {variation.shape}"
valid_mask = ~np.isnan(variation).any(axis=1)  
print(f"variation shape: {variation.shape}, mask shape: {valid_mask.shape}", file=sys.stderr)
variation = variation[valid_mask]
valid_symbols = np.array(valid_symbols)[valid_mask]
valid_names   = np.array(valid_names)[valid_mask]
 
print(f"Stocks after NaN removal: {len(valid_names)}", file=sys.stderr)

from sklearn import covariance
print("\nFitting Graphical Lasso …", file=sys.stderr)
alphas = np.logspace(-1.5, 1, num=10)
edge_model = covariance.GraphicalLassoCV(alphas=alphas)

#Standardize the time series
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

from sklearn import cluster
print("Clustering with Affinity Propagation …", file=sys.stderr)
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
n_labels = labels.max()
print(f"\nFound {n_labels} clusters:\n")
for i in range(n_labels + 1):
    print(f"Cluster {i + 1}: {', '.join(valid_names[labels == i])}")

from sklearn import manifold
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver="dense", n_neighbors=6
    )
embedding = node_position_model.fit_transform(X.T).T

from matplotlib.collections import LineCollection

plt.figure(1, facecolor='w', figsize=(15,8))
plt.clf()
ax = plt.axes([0.0, 0.0, 1.0, 1.0])
plt.axis('off')

partial_correlations = edge_model.precision_.copy()
d = 1/np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

# Plot the nodes using the coordinates of our embedding

plt.scatter(
    embedding[0], embedding[1], s=100 * d**2, c=labels, cmap=plt.cm.nipy_spectral
    )
# Plot the edges
start_idx, end_idx = non_zero.nonzero()
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [
    [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(
    segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
)
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = "left"
        x = x + 0.002
    else:
        horizontalalignment = "right"
        x = x - 0.002
    if this_dy > 0:
        verticalalignment = "bottom"
        y = y + 0.002
    else:
        verticalalignment = "top"
        y = y - 0.002
    plt.text(
        x,
        y,
        name,
        size=10,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        bbox=dict(
            facecolor="w",
            edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            alpha=0.6,
        ),
    )

plt.xlim(
    embedding[0].min() - 0.15 * np.ptp(embedding[0]),
    embedding[0].max() + 0.10 * np.ptp(embedding[0]),
)
plt.ylim(
    embedding[1].min() - 0.03 * np.ptp(embedding[1]),
    embedding[1].max() + 0.03 * np.ptp(embedding[1]),
)
plt.title(
    f"Saudi Stock Market — Affinity Propagation Clusters ({START} – {END} - Author: Ericka Barbosa)",
    fontsize=13,
    pad=20,
)
output_path = "C:/Users/Ericka/Desktop/Python/saudi_stock_clusters.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved → {output_path}", file=sys.stderr)
plt.show()
plt.show()

