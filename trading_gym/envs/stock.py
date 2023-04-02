from concurrent import futures
from typing import Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas.plotting import register_matplotlib_converters

from ..type import Action
from ..utils import Indicators, OrderHandler, get_logger

logger = get_logger(__name__)
register_matplotlib_converters()
INF = 1e10


class StocksEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_shares: int = 1000,
        initial_balance: float = 10_000.0,
        plot_every: int = 100,
        video_path: Optional[str] = "~/Downloads/trading_gym.mp4",
    ):
        """
        Stock trading environment
        :param df: DataFrame with stock prices and volume in OHLCV format
        """
        super().__init__()
        self.df = df
        self.max_shares = max_shares
        self._process_data()
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=4)
        self.plot_every = plot_every  # plot every n episodes
        self.video_path = video_path

        self.action_space = spaces.Discrete(len(Action))  # Buy, Sell, or Hold
        self.observation_space = spaces.Dict(
            {
                "balance": spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
                "equity": spaces.Box(low=0, high=INF, shape=(1,), dtype=np.uint32),
                "features": spaces.Box(low=0, high=INF, shape=(8,), dtype=np.float64),
            }
        )

        # episode
        self._episode = 0
        self._plots: list[plt.Figure] = []
        self._orders = OrderHandler()
        self._init_balance = initial_balance
        self._balance = self._last_balance = self._init_balance
        self._total_reward = 0.0
        self._start_tick = 0
        self._end_tick = len(self.df) - 1
        self._done = False
        self._current_tick = self._start_tick
        self._total_reward_history = np.zeros(len(self.df), dtype=np.float64)
        self._portfolio_values = np.array(
            [self._balance] * len(self.df), dtype=np.float64
        )

    def _process_data(self):
        """Process the data"""
        # try to set the index to date
        if "Date" in self.df.columns:
            self.df = self.df.set_index("Date")
        # Remove timezone from the index
        self.df.index = self.df.index.tz_localize(None)

        # add volume if it doesn't exist
        if "Volume" not in self.df.columns:
            self.df["Volume"] = 0

        # add indicators dataframe with the self.df
        ind = Indicators(self.df).add_all()
        self.df = self.df.join(ind).dropna()

    @property
    def _qtn(self) -> int:
        """Get the quantity of shares"""
        return self._orders.positions

    @property
    def _equity(self) -> float:
        """Get the current equity. The unrealized profit"""
        if self._qtn == 0 or self._done:
            return 0
        tax = self._orders.calc_tax(self._current_price, self._qtn, Action.SELL)
        return (self._qtn * self._current_price) - tax

    @property
    def _obs(self):
        """Get the observation"""
        return {
            "balance": np.array([self._balance]),
            "equity": np.array([self._equity]),
            "quantity": np.array([self._qtn]),
            "features": self.df.iloc[self._current_tick].to_numpy(),
        }

    @property
    def _info(self):
        """Get the info"""
        return {
            "total_reward": self._total_reward,
        }

    @property
    def _current_price(self):
        """Get the current price"""
        return self.df["Close"].iloc[self._current_tick]

    @property
    def _current_date(self):
        """Get the current date"""
        return self.df.index[self._current_tick]

    @property
    def _last_action(self) -> Optional[int]:
        """Get the last order action"""
        if self._orders.latest_order:
            return self._orders.latest_order[1]
        return None

    @property
    def _last_qtn(self) -> Optional[int]:
        """Get the last order quantity"""
        if self._orders.latest_order:
            return self._orders.latest_order[3]
        return None

    def _get_reward(self, action: Action, fee: float) -> float:
        """Get the reward for the current tick"""
        reward = 1e-5
        # Keep track of the history of portfolio values
        current_value = self._balance + self._equity
        self._portfolio_values[self._current_tick] = current_value
        if action == Action.SELL:
            reward += self._orders.latest_profit / (self._current_price * 100)
        elif action == Action.BUY:
            reward -= fee / (self._current_price * 100)

        if reward < 0:
            # Penalize the agent for selling at a loss
            reward *= 0.5
        # Update the env variables
        self._total_reward += reward
        if (
            self._orders.latest_order
            and self._orders.latest_order[0] == self._current_tick
        ):
            return reward
        self._total_reward_history[self._current_tick] = self._total_reward

        return reward

    def step(self, action):
        """Take a step in the environment"""
        # We are done if we blow up our balance by 50% or if we reach the end of the data
        if ((self._balance <= (self._init_balance * 0.5)) and (self._equity <= 0)) or (
            self._current_tick >= self._end_tick
        ):
            self._done = True

        # Get the trade cost and fee
        cost, fee = self._orders.add(action, self._current_price, self._current_date)
        total_cost = round(cost + fee, 2)
        if action != Action.HOLD:
            # We are increasing the quantity for the last trade
            if self._last_action == action:
                self._balance = self._last_balance - total_cost
            else:
                # We are making a new trade on a new day
                self._last_balance = self._balance
                self._balance -= total_cost

        reward = self._get_reward(action, fee)
        observation = self._obs
        info = self._info

        if self._current_tick < self._end_tick and action == Action.HOLD:
            # Move to the next tick
            self._current_tick += 1

        return observation, reward, self._done, info

    def reset(self, seed=None, options=None):
        """Reset the environment data and state"""
        super().reset(seed=seed, options=options)

        self._episode += 1
        self._balance = self._last_balance = self._init_balance
        self._total_reward = 0.0
        self._done = False
        self._current_tick = self._start_tick
        self._portfolio_values[:] = self._balance
        self._total_reward_history[:] = 0.0
        self._plots.clear()
        self._orders.reset()

        observation = self._obs
        info = self._info
        return observation, info

    def render(self, mode="human"):
        """Render the stock chart with the current position"""
        if mode != "human":
            return

        if not self._done:
            df = self.df.iloc[: self._current_tick]
            portfolio = self._portfolio_values[: self._current_tick]
            rewards = self._total_reward_history[: self._current_tick]
        else:
            df = self.df
            portfolio = self._portfolio_values
            rewards = self._total_reward_history
        chunk_size = len(df) // 200
        chunks = [
            df.iloc[i : i + len(df) // chunk_size]
            for i in range(0, len(df), len(df) // chunk_size)
        ]
        futs = []
        for chunk in chunks:
            start_row = chunk.index[0]
            end_row = chunk.index[-1]
            start_idx = df.index.get_loc(start_row)
            end_idx = df.index.get_loc(end_row)
            # Return when chunk is too small
            if end_idx - start_idx < 10:
                continue
            portfolio_values_chunk = portfolio[start_idx : end_idx + 1]
            reward_chunk = rewards[start_idx : end_idx + 1]
            # self._draw_plot(chunk, portfolio_values_chunk, reward_chunk)
            futs.append(
                self._thread_pool.submit(
                    self._draw_plot, chunk, portfolio_values_chunk, reward_chunk
                )
            )

        # Wait for all plots to be drawn
        for fut in futs:
            fut.result()

        # Render the plot
        for p in self._plots:
            p.show()

    def _draw_plot(
        self, df: pd.DataFrame, portfolio_chunk: np.ndarray, reward_chunk: np.ndarray
    ):
        """Draw the plot of the stock chart with the current position"""

        # Define plot styling options
        mpl_style = mpf.make_mpf_style(
            base_mpl_style="seaborn-darkgrid", facecolor="lightgrey", gridcolor="white"
        )

        # Plot candlestick chart with OHLCV data and overlay EMA, SMA, and Volume
        plot_data = [
            mpf.make_addplot(df["EMA"], color="mediumseagreen"),
            mpf.make_addplot(df["SMA"], color="cornflowerblue"),
            mpf.make_addplot(df["RSI"], panel=2, color="mediumorchid"),
            mpf.make_addplot(
                portfolio_chunk, panel=3, color="deepskyblue", ylabel="Portfolio Value"
            ),
            mpf.make_addplot(
                reward_chunk, panel=4, color="dodgerblue", ylabel="Total Reward"
            ),
            mpf.make_addplot([30] * len(df), panel=2, color="r", type="line"),
            mpf.make_addplot([70] * len(df), panel=2, color="r", type="line"),
        ]

        # Mark buy and sell orders in the candlestick chart
        buy_orders = self._orders.get(int(Action.BUY), df)
        sell_orders = self._orders.get(int(Action.SELL), df)
        # get high value for buy orders and low value for sell orders and nan if no order
        if buy_orders.shape[0]:
            # Get those buy orders which are not in the current chunk
            buy_orders = df.loc[buy_orders, "High"].reindex(df.index).values
            plot_data.append(
                mpf.make_addplot(
                    buy_orders,
                    type="scatter",
                    marker="^",
                    markersize=200,
                    color="green",
                    panel=0,
                )
            )
        if sell_orders.shape[0]:
            sell_orders = df.loc[sell_orders, "High"].reindex(df.index).values
            plot_data.append(
                mpf.make_addplot(
                    sell_orders,
                    type="scatter",
                    marker="v",
                    markersize=200,
                    color="red",
                    panel=0,
                )
            )

        # Define figure size and subplot ratios
        fig, axs = mpf.plot(
            df,
            style=mpl_style,
            type="candle",
            volume=True,
            addplot=plot_data,
            returnfig=True,
            panel_ratios=(4, 1, 1, 1, 1),
            figsize=(10, 9),
        )

        self._plots.append(fig)
        logger.info("Plot drawn for episode %s", self._episode)

    def _export_plot2vid(self):
        """Export the plots to a video"""
        if not self._plots:
            logger.info("No plots to export")
            return
        import matplotlib.animation as animation

        def plot2image(plot: plt.Figure) -> np.ndarray:
            plot.canvas.draw()
            # extract the image data from the plot object
            w, h = plot.canvas.get_width_height()
            return np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                (h, w, 3)
            )

        # create a writer object that outputs to a BytesIO buffer
        writer = animation.FFMpegWriter(fps=2, extra_args=["-threads", "4"])
        # Save the video to a file
        with writer.saving(
            plt.figure(figsize=self._plots[0].get_size_inches()),
            self.video_path,
            dpi=100,
        ):
            # iterate over the list of plots and add them to the animation
            for p in self._plots:
                plt.clf()
                plt.gca().imshow(plot2image(p))
                writer.grab_frame()
        logger.info("Video exported to %s", self.video_path)

    def close(self):
        """Close the environment"""
        self.reset()
        self._done = True
        self.df = None
        self._orders.conn.close()

    def legal_actions(self):
        """Get the legal actions for the current tick"""
        actions = []
        if (
            self._balance > 0
            and self._last_action != Action.SELL
            and min(self.max_shares, self._balance // self._current_price) > 0
        ):  # Buy action
            actions.append(int(Action.BUY))

        if self._qtn > 0 and self._last_action != Action.BUY:
            # Sell action
            actions.append(int(Action.SELL))

        actions.append(int(Action.HOLD))  # Hold action
        return actions

    @staticmethod
    def action_to_string(action_number: Literal[0, 1, 2]):
        """Convert an action number to a string"""
        return str(Action._value2member_map_[action_number])
