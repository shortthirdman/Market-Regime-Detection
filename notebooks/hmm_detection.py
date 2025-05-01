import backtrader as bt
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
from hmmlearn import hmm
import matplotlib.pyplot as plt
# Optional: Configure matplotlib backend if needed
%matplotlib qt5

def train_hmm_and_identify_states(df, n_states=5, n_iter=500, tol=1e-4, vol_window=20):
    """
    Train an HMM on [Log Return, Volatility of Log Return], label each bar with its state,
    and identify strong/weak bull & bear plus ranging regimes based on mean log return.
    # ... (docstring continues) ...
    """
    df_hmm = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # --- Feature Calculation using Log Returns ---
    # Log returns are often preferred in finance as they are additive over time
    # and approximate percentage changes for small values.
    df_hmm['Log Return'] = np.log(df_hmm['Close'] / df_hmm['Close'].shift(1))
    df_hmm['Log Return'].fillna(0, inplace=True) # Handle the first NaN value

    # Calculate rolling standard deviation of log returns as a measure of volatility
    df_hmm['Volatility'] = df_hmm['Log Return'].rolling(vol_window).std()
    df_hmm['Volatility'].fillna(0, inplace=True) # Handle initial NaNs from rolling window

    # --- Select features for HMM ---
    # The HMM will learn hidden states based on these observable features.
    # More features could potentially improve state differentiation.
    X = df_hmm[['Log Return', 'Volatility']].values

    # Handle potential numerical issues before fitting
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: NaNs or Infs detected in HMM features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- HMM Training ---
    # GaussianHMM assumes the features within each hidden state follow a Gaussian distribution.
    # 'n_components': The number of hidden states to find (a key tuning parameter).
    # 'covariance_type="diag"': Assumes features are independent within a state (simpler, less prone to overfitting).
    # 'n_iter', 'tol': Control the convergence of the training algorithm.
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        tol=tol,
        random_state=42, # For reproducibility
        verbose=False
    )

    print(f"\nFitting HMM with {n_states} states...")
    try:
        # Fit the HMM model to the feature data (X)
        with warnings.catch_warnings(): # Suppress specific warnings during fitting
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X)
    except ValueError as e:
        print(f"Error fitting HMM: {e}")
        print("Check input data X for issues.")
        raise e

    if not model.monitor_.converged:
        print(f"Warning: HMM did not converge after {n_iter} iterations.")

    # Predict the most likely hidden state for each data point
    states = model.predict(X)
    df_hmm['HMM_State'] = states # Add the predicted states back to the DataFrame

    # --- State Interpretation ---
    # Analyze the characteristics of each predicted state
    stats = []
    for i in range(n_states):
        mask = (states == i)
        if mask.sum() == 0: # Check if a state was even predicted
            print(f"Warning: State {i} was not predicted for any data point.")
            continue
        # Calculate average log return and volatility for data points belonging to this state
        stats.append({
            'State': i,
            'Mean Log Return': df_hmm.loc[mask, 'Log Return'].mean(),
            'Mean Volatility': df_hmm.loc[mask, 'Volatility'].mean(),
            'Count': mask.sum() # How many data points belong to this state
        })

    if not stats:
        raise ValueError("HMM training resulted in no predictable states.")

    # Sort states by their average log return (descending)
    # Assumption: Highest mean return = Strong Bull, Lowest mean return = Strong Bear
    stats_df = pd.DataFrame(stats).sort_values('Mean Log Return', ascending=False).reset_index(drop=True)

    print("\nHMM State Summary (sorted by Mean Log Return):")
    print(stats_df.to_string(index=False, float_format='{:.6f}'.format))

    # Assign regimes based on sorted order (assuming 5 states initially)
    state_indices = stats_df['State'].tolist()
    s_bull_strong = -1 # Initialize with invalid index
    s_bear_strong = -1

    # Adjust assignment based on how many distinct states were actually found
    if len(state_indices) > 0:
        s_bull_strong = state_indices[0]      # State with highest mean log return
        s_bear_strong = state_indices[-1]     # State with lowest mean log return
    # (The code handles cases with < 5 states by only assigning strong bull/bear)

    print(f"\nRegime mapping (based on Mean Log Return sort):")
    print(f"  Strong Bull State = {s_bull_strong} (Highest Mean Log Return)")
    print(f"  Strong Bear State = {s_bear_strong} (Lowest Mean Log Return)")

    # Basic check for valid state assignment
    if s_bull_strong < 0 or s_bear_strong < 0:
          print("\nError: Could not reliably assign Strong Bull or Strong Bear state index.")
          # The indicator initialization will later catch these invalid indices

    print("\nReturning states for Strong Bull and Strong Bear signals.")
    # Return the DataFrame with HMM states and the identified indices for strong bull/bear
    return df_hmm, s_bull_strong, s_bear_strong

class HMMData(bt.feeds.PandasData):
    """Custom PandasData that carries the HMM_State column through as `hmm_state`."""
    lines = ('hmm_state',) # Declare the new data line
    params = (
        # Map standard OHLCV columns
        ('datetime', None), # Use index for datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None), # Not used here
        # Map our custom column 'HMM_State' from the DataFrame to the 'hmm_state' line
        ('hmm_state', 'HMM_State'),
    )


class HMMRegimeStartSignal(bt.Indicator):
    """
    Signals the first bar of each new strong bull or strong bear regime
    by comparing the current HMM state to the prior bar.
    """
    lines = ('bull_start', 'bear_start',) # Output lines for signals
    params = (
        ('bull_state_idx', None), # Parameter to receive the strong bull state index
        ('bear_state_idx', None), # Parameter to receive the strong bear state index
    )
    plotinfo = dict(subplot=False) # Plot directly on the price chart
    plotlines = dict(
        # Define how the signals should be plotted (green up triangles, red down triangles)
        bull_start=dict(marker='^', markersize=8, color='green', linestyle='None'),
        bear_start=dict(marker='v', markersize=8, color='red',   linestyle='None'),
    )

    def __init__(self):
        # Validate that valid state indices were passed from the main script
        if self.p.bull_state_idx is None or self.p.bear_state_idx is None or \
           self.p.bull_state_idx < 0 or self.p.bear_state_idx < 0:
            raise ValueError("Must pass valid non-negative bull_state_idx and bear_state_idx to HMMRegimeStartSignal")
        # Access the custom hmm_state line from the data feed
        self.hmm_state = self.data.hmm_state

    def next(self):
        # Called for each bar of data (once enough data is available)
        if len(self.data) < 2: # Need at least two bars to compare current and previous state
            return

        # Default signal values to NaN (no signal)
        self.lines.bull_start[0] = float('nan')
        self.lines.bear_start[0] = float('nan')

        # Get current and previous HMM state
        curr = int(self.data.hmm_state[0])
        prev = int(self.data.hmm_state[-1])
        b = self.p.bull_state_idx # Convenience alias for bull state index
        r = self.p.bear_state_idx # Convenience alias for bear state index (renamed from 'r' for clarity)

        # --- Signal Logic ---
        # Strong bull entry: Current state is strong bull, previous was not.
        if curr == b and prev != b:
            # Place a green marker slightly below the low of the current bar
            self.lines.bull_start[0] = self.data.low[0] * 0.99

        # Exit strong bull: Previous state was strong bull, current is not.
        # This is treated as a potential sell/bearish signal.
        elif prev == b and curr != b:
            # Place a red marker slightly above the high of the current bar
            self.lines.bear_start[0] = self.data.high[0] * 1.01

        # Strong bear entry: Current state is strong bear, previous was not.
        elif curr == r and prev != r:
            # Place a red marker slightly above the high of the current bar
            self.lines.bear_start[0] = self.data.high[0] * 1.01


if __name__ == '__main__':
    ticker, start, end = 'BTC-USD', '2022-01-01', '2023-12-31'

    print(f"\nDownloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    # Optional: Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # --- Run HMM ---
    # Call the function to train HMM and get the state-augmented data + regime indices
    data_with_hmm, bull_state, bear_state = train_hmm_and_identify_states(df)

    # --- Backtrader Setup ---
    cerebro = bt.Cerebro(stdstats=False) # Create the main backtrader engine instance

    # --- Add Data ---
    # Ensure DataFrame index is DatetimeIndex (usually true for yfinance)
    if not isinstance(data_with_hmm.index, pd.DatetimeIndex):
         data_with_hmm.index = pd.to_datetime(data_with_hmm.index)
    # Create the custom data feed using the HMM-augmented DataFrame
    data_feed = HMMData(dataname=data_with_hmm)
    cerebro.adddata(data_feed) # Add the data feed to Cerebro

    # --- Add Indicators ---
    # Add the custom HMM signal indicator, passing the identified state indices
    cerebro.addindicator(HMMRegimeStartSignal,
                         bull_state_idx=bull_state,
                         bear_state_idx=bear_state)
    # Add standard Moving Average indicators for context
    cerebro.addindicator(bt.indicators.SimpleMovingAverage, period=30)
    cerebro.addindicator(bt.indicators.SimpleMovingAverage, period=90)

    # --- Run and Plot ---
    print("\n--- Running Cerebro (for plotting) ---")
    cerebro.run() # Run the engine (calculates indicators)

    # Configure plot appearance
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    print("\n--- Generating Plot ---")
    # Generate the plot: includes price, volume, SMAs, and HMM signals
    cerebro.plot(style='line', volume=True, iplot=False) # iplot=False for static plot