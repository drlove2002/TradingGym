import os
import pandas as pd

ROOT_DIR = "/home/love/Documents/TradingGym/trading_gym"
data_path = ROOT_DIR + "/data/reliance.csv"
#%%
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
    df.to_csv(data_path)
else:
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
#%%
df = df.drop(columns=["Dividends", "Stock Splits"])
#%%
from trading_gym.envs.stock import StocksEnv
env = StocksEnv(df=df, window_size=10)
#%%
env.step((0, 1))
#%%
env.step((1, env._qtn))
#%%
env.step((0, 0))
#%%
env.render()