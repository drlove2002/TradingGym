import os
import pandas as pd

ROOT_DIR = "/home/love/Documents/TradingGym/trading_gym"
data_path = ROOT_DIR + "/data/reliance.csv"
# %%
# Download data if not present
if not os.path.exists(data_path):
    import yfinance as yf

    # Ticker of reliance industries
    ticker = yf.Ticker("RELIANCE.NS")
    df = ticker.history(
        # period="max",
        period="2y",
        interval="1d",
    )
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df.to_csv(data_path)
else:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
# %%
from trading_gym.envs.stock import StocksEnv

env = StocksEnv(df=df, window_size=10)
# %%
env.observation_space.sample()
# %%
import numpy as np
# Get observation space as a 1d array also flatten the dict
np.concatenate(list(env.observation_space.sample().values()))
# %%
env.step(0)
# %%
env.step(1)
# %%
env.step(2)
# %%
env.render()
# %%
env.reset()
# %%
env.close()
# TODO : Fix reward calculation
# TODO : Add video recording after every episode
# TODO : Add plotting of entire stock data with buy and sell points