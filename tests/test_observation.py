import os
import pandas as pd
import numpy as np

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
env = StocksEnv(df=df, max_shares=100, video_path="trading_gym/data/animation.mp4")
# %%
while not env._done:
    action = np.random.choice(env.legal_actions())
    env.step(action)
# %%
env._plots[0].show()
# %%
env.reset()
# %%
env.close()
# %%
env._orders.reset()
# %%
env.render()
# TODO: Bug in selling amd buying price and profit calculation
# TODO: Fix reward calculation