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
env = StocksEnv(df=df)
# %%
done = False
while not done:
    action = np.random.choice(env.legal_actions())
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
# %%
env.step(2)
# %%
env._plots[1].show()
# %%
env.reset()
# %%
env.close()
# %%
env.render()
