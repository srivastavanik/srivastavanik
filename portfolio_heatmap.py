import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.widgets as widgets
import datetime

# Step 1: Load Portfolio Data
def load_portfolio(file_path):
    "Load the portfolio data from CSV"
    return pd.read_csv(file_path)

# Step 2: Fetch Historical Price Data
def fetch_price_data(tickers, start_date, end_date):
    "Fetch historical price data using yfinance"
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Step 3: Fetch Real-Time Price Data
def fetch_real_time_price_data(tickers):
    "Fetch real-time price data using yfinance"
    data = yf.download(tickers, period="1d", interval="1m")['Adj Close']
    return data

# Step 4: Calculate Returns
def calculate_returns(price_data):
    "Calculate daily returns from price data"
    returns = price_data.pct_change(fill_method=None).dropna()
    return returns

# Step 5: Compute Sharpe Ratio
def calculate_sharpe_ratios(returns, risk_free_rate=0.01):
    "Calculate Sharpe Ratio for each asset"
    excess_returns = returns.mean() - risk_free_rate / 252
    volatility = returns.std()
    sharpe_ratios = excess_returns / volatility
    return sharpe_ratios

def plot_sharpe_ratios(sharpe_ratios):
    "Plot the Sharpe Ratios for each asset"
    plt.figure(figsize=(14, 8))
    sharpe_ratios.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Sharpe Ratio for Each Asset', fontsize=20)
    plt.xlabel('Assets', fontsize=16)
    plt.ylabel('Sharpe Ratio', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Step 6: Rolling Correlation Heatmap with Time Slider
def plot_rolling_correlation_with_slider(returns, window=60):
    "Plot rolling correlation heatmap with a time slider to show changes over time"
    rolling_corr = returns.rolling(window=window).corr().dropna()

    # Extract unique dates for slider functionality
    unique_dates = rolling_corr.index.get_level_values(0).unique()

    fig, ax = plt.subplots(figsize=(14, 8))
    slider_ax = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    time_slider = widgets.Slider(slider_ax, 'Time', 0, len(unique_dates) - 1, valinit=0, valfmt='%0.0f')

    def update(val):
        ax.clear()
        time_index = int(time_slider.val)
        current_corr = rolling_corr.xs(unique_dates[time_index])
        sns.heatmap(current_corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(f'Rolling Correlation Heatmap (Date: {unique_dates[time_index]})', fontsize=16)
        ax.set_xlabel('Assets', fontsize=14)
        ax.set_ylabel('Assets', fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

    time_slider.on_changed(update)
    update(0)
    plt.show()

# Step 7: Perform Clustering
def perform_clustering(correlation_matrix, n_clusters=4):
    "Perform K-Means clustering on the correlation matrix"
    # Convert correlation matrix to distance matrix
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(distance_matrix)
    return clusters

# Step 8: Cluster Risk Heatmap
def plot_cluster_risk_heatmap(portfolio, clusters, returns):
    "Extend clustering analysis to include portfolio weights and standard deviation of returns"
    portfolio['Cluster'] = clusters
    portfolio['Volatility'] = returns.std()
    cluster_risk = portfolio.groupby('Cluster')[['Weight', 'Volatility']].mean()

    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_risk, annot=True, cmap='coolwarm', linewidths=0.5, cbar=False, annot_kws={'size': 12})
    plt.title('Cluster Risk Heatmap', fontsize=20)
    plt.xlabel('Metrics', fontsize=16)
    plt.ylabel('Clusters', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

# Step 9: Volatility Analysis
def plot_volatility_analysis(returns):
    "Plot volatility analysis for each asset"
    volatility = returns.std()
    if not volatility.empty:
        plt.figure(figsize=(14, 8))
        volatility.sort_values().plot(kind='bar', color='coral', edgecolor='black')
        plt.title('Volatility of Each Asset', fontsize=20)
        plt.xlabel('Assets', fontsize=16)
        plt.ylabel('Volatility (Standard Deviation)', fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Volatility vs Return Scatterplot
    average_returns = returns.mean()
    plt.figure(figsize=(12, 8))
    plt.scatter(volatility, average_returns, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Volatility vs Return Scatterplot', fontsize=20)
    plt.xlabel('Volatility (Standard Deviation)', fontsize=16)
    plt.ylabel('Average Daily Return', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

# Step 10: Asset Contribution to Portfolio Variance
def plot_contribution_to_risk(returns, weights):
    "Plot each asset's contribution to portfolio variance"
    cov_matrix = returns.cov()
    portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contributions = cov_matrix.dot(weights) / portfolio_volatility
    risk_contributions = weights * marginal_contributions

    plt.figure(figsize=(14, 8))
    risk_contributions.sort_values().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Contribution to Portfolio Risk by Asset', fontsize=20)
    plt.xlabel('Assets', fontsize=16)
    plt.ylabel('Contribution to Portfolio Variance', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Step 11: Correlation Network Graph
def plot_correlation_network(returns, threshold=0.5):
    "Create a correlation network graph with edges representing correlations"
    correlation_matrix = returns.corr()
    graph = nx.Graph()

    # Add nodes and edges above threshold
    for i, asset_i in enumerate(correlation_matrix.columns):
        graph.add_node(asset_i)
        for j, asset_j in enumerate(correlation_matrix.columns):
            if i < j and abs(correlation_matrix.iloc[i, j]) > threshold:
                graph.add_edge(asset_i, asset_j, weight=correlation_matrix.iloc[i, j])

    # Animation function for the network graph
    pos = nx.spring_layout(graph, seed=42, iterations=50)
    fig, ax = plt.subplots(figsize=(14, 10))

    def update(frame):
        ax.clear()
        pos_frame = nx.spring_layout(graph, seed=42, iterations=frame+1)
        nx.draw_networkx(graph, pos_frame, with_labels=True, node_size=800, node_color='skyblue', font_size=12, font_weight='bold', edge_color='lightgray', ax=ax)
        ax.set_title('Correlation Network Graph', fontsize=20)
        ax.axis('off')
    
    ani = FuncAnimation(fig, update, frames=range(50), repeat=True)
    plt.show()

# Main Function
if __name__ == "__main__":
    # File Paths
    portfolio_file = "portfolio.csv"
    
    # Step 1: Load Portfolio Data
    portfolio = load_portfolio(portfolio_file)
    
    # Step 2: Fetch Historical Price Data
    tickers = portfolio['Ticker'].tolist()
    price_data = fetch_price_data(tickers, start_date="2020-01-01", end_date=datetime.datetime.now().strftime('%Y-%m-%d'))
    
    # Step 3: Calculate Returns
    returns = calculate_returns(price_data)

    # Step 4: Sharpe Ratios
    sharpe_ratios = calculate_sharpe_ratios(returns)
    plot_sharpe_ratios(sharpe_ratios)

    # Step 5: Rolling Correlation Heatmap with Slider
    plot_rolling_correlation_with_slider(returns)

    # Step 6: Perform Clustering
    correlation_matrix = returns.corr()
    clusters = perform_clustering(correlation_matrix, n_clusters=4)
    plot_cluster_risk_heatmap(portfolio, clusters, returns)

    # Step 7: Volatility Analysis
    plot_volatility_analysis(returns)

    # Step 8: Contribution to Portfolio Risk
    weights = portfolio['Weight'].values
    plot_contribution_to_risk(returns, weights)

    # Step 9: Fetch and Plot Real-Time Data
    real_time_data = fetch_real_time_price_data(tickers)
    real_time_returns = calculate_returns(real_time_data)
    plot_sharpe_ratios(calculate_sharpe_ratios(real_time_returns))

    # Step 10: Animated Correlation Network Graph
    plot_correlation_network(returns, threshold=0.5)