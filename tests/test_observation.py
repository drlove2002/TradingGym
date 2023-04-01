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
from trading_gym.envs.stock import StocksEnv, Action
env = StocksEnv(df=df, max_shares=100, video_path="trading_gym/data/animation.mp4")
# %%
# while env._done is False:
#     action = np.random.choice(env.legal_actions())
#     env.step(action)
step = env.step(1)
legal = env.legal_actions()
# %%
env._plots[3].show()
# for p in env._plots:
#     p.show()
# %%
env.reset()
# %%
env.close()
# %%
env._orders.reset()
# %%
chunk_size = 4
chunks = [env.df.iloc[i:i + len(env.df) // chunk_size] for i in range(0, len(env.df), len(env.df) // chunk_size)]
for chunk in chunks:
    start_row = chunk.index[0]
    end_row = chunk.index[-1]
    start_idx = env.df.index.get_loc(start_row)
    end_idx = env.df.index.get_loc(end_row)
    # Return when chunk is too small
    if end_idx - start_idx < 10:
        continue
    portfolio_values_chunk = env._portfolio_values[start_idx:end_idx+1]
    reward_chunk = env._total_reward_history[start_idx:end_idx+1]
    env._draw_plot(chunk, portfolio_values_chunk, reward_chunk)
# TODO: Bug in selling amd buying price and profit calculation
# TODO: Fix reward calculation
# TODO: Find a way to save the plots in folders so that we can make animation