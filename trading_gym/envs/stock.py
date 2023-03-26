from collections import deque
from concurrent import futures
from typing import Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas.plotting import register_matplotlib_converters

from ..type import Action
from ..utils import Indicators, OrderHandler

register_matplotlib_converters()
INF = 1e10


class StocksEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        max_shares: int = 1000,
        indicator_factory: Indicators = None,
    ):
        """
        Stock trading environment
        :param df: DataFrame with stock prices and volume in OHLCV format
        :param window_size: Number of days in the past to consider
        :param indicator_factory: Indicator factory to add indicators to the data
        """
        super().__init__()
        self.df = df
        self._executor = futures.ThreadPoolExecutor(max_workers=4)
        self.window_size = window_size
        self.max_shares = max_shares
        self._process_data(indicator_factory or Indicators(self.df))

        self.action_space = spaces.Discrete(len(Action))  # Buy, Sell, or Hold
        self.observation_space = spaces.Dict(
            {
                "balance": spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float64),
                "equity": spaces.Box(low=0, high=INF, shape=(1,), dtype=np.uint32),
                "features": spaces.Box(low=0, high=INF, shape=(8,), dtype=np.float64),
            }
        )

        # episode
        self._orders = OrderHandler()
        self._balance = 10_000.0
        self._last_balance = 10_000.0
        self._total_reward = 0.0
        self._start_tick = self.window_size
        self._end_tick = len(self.df) - 1
        self._done = False
        self._current_tick = self._start_tick
        self._total_reward_history = deque([0.0] * window_size, maxlen=window_size)
        self._portfolio_values = deque(
            [self._balance + self._equity] * window_size, maxlen=window_size
        )
        self._plots: list[plt.Figure] = []
        self._draw_plot()

    def _process_data(self, indicators: Indicators):
        """Process the data"""
        # try to set the index to date
        if "Date" in self.df.columns:
            self.df = self.df.set_index("Date")

        # add volume if it doesn't exist
        if "Volume" not in self.df.columns:
            self.df["Volume"] = 0

        # add indicators dataframe with the self.df
        self.df = self.df.join(indicators.add_all()).dropna()

    @property
    def _features(self):
        """Get the features for the current tick. This is the data that will be fed to the model"""
        return self.df[
            self._current_tick - self.window_size : self._current_tick
        ].copy()

    @property
    def _qtn(self) -> int:
        """Get the quantity of shares"""
        return self._orders.positions

    @property
    def _equity(self) -> float:
        """Get the current equity. The unrealized profit"""
        tax = self._orders.calc_tax(self._current_price, self._qtn, Action.SELL)
        return (self._qtn * self._current_price) - tax

    @property
    def _obs(self):
        """Get the observation"""
        return {
            "balance": np.array([self._balance]),
            "equity": np.array([self._equity]),
            "quantity": np.array([self._qtn]),
            "features": self._features.iloc[-1].values,
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
            return self._orders.latest_order[4]
        return None

    def _get_reward(self, action: Action, fee: float) -> float:
        """Get the reward for the current tick"""
        reward = 0.0
        # Keep track of the history of portfolio values
        current_value = self._balance + self._equity
        self._portfolio_values.append(current_value)
        if action == Action.SELL:
            reward += self._orders.latest_profit
        elif action == Action.BUY:
            reward -= fee

        # Update the env variables
        self._total_reward += reward
        self._total_reward_history.append(self._total_reward)

        return reward

    def step(self, action):
        """Take a step in the environment"""
        last_action = self._last_action
        # Get the trade cost and fee
        cost, fee = self._orders.add(action, self._current_price, self._current_date)
        total_cost = round(cost + fee, 2)
        if action != Action.HOLD:
            # We are increasing the quantity for the last trade
            if last_action == action:
                self._balance = self._last_balance - total_cost
            else:
                # We are making a new trade on a new day
                self._last_balance = self._balance
                self._balance -= total_cost

        # Check if we are done
        if (self._balance <= 0 and self._equity <= 0) or (
            self._current_tick >= self._end_tick
        ):
            self._done = True

        reward = self._get_reward(action, fee)
        observation = self._obs
        info = self._info
        self._executor.submit(self._draw_plot)

        if self._current_tick < self._end_tick and action == Action.HOLD:
            # Move to the next tick
            self._current_tick += 1

        return observation, reward, self._done, info

    def reset(self, seed=None, options=None):
        """Reset the environment data and state"""
        super().reset(seed=seed, options=options)

        self._executor.submit(self._export_plot2vid)
        self._balance = self._last_balance = 10_000.0
        self._total_reward = 0.0
        self._done = False
        self._current_tick = self._start_tick
        self._portfolio_values.extend([self._balance] * self.window_size)
        self._total_reward_history.extend([0.0] * self.window_size)
        self._plots.clear()
        self._draw_plot()
        self._orders.reset()

        observation = self._obs
        info = self._info
        return observation, info

    def render(self, mode="human"):
        """Render the stock chart with the current position"""
        self._plots[-1].show()

    def _draw_plot(self):
        """Draw the plot of the stock chart with the current position"""
        df_chunk: pd.DataFrame = self._features

        # Create a function to color the candlesticks based on their direction
        def color_candlestick(candle):
            return "g" if candle["Close"] > candle["Open"] else "r"

        # Create the candlestick chart using matplotlib
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(10, 10),
            facecolor="white",
            sharex=True,  # type: ignore
            gridspec_kw={"height_ratios": [3, 1, 1, 1]},
        )
        fig: plt.Figure
        plt.suptitle("Stock Market Trading Environment", fontsize=16)
        plt.xlabel("Date")

        # Plot the candlesticks and color them based on their direction
        ax1.set_title("Candlestick Chart")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        for index, row in df_chunk.iterrows():
            color = color_candlestick(row)
            # Remove the time part because we'll only be using the date
            ax1.plot(
                [index, index],
                [row["Low"], row["High"]],
                color=color,
                linewidth=((5 * 10) / self.window_size),
            )
            ax1.plot(
                [index, index],
                [row["Open"], row["Close"]],
                color=color,
                linewidth=((25 * 10) / self.window_size),
            )

        # Plot the EMA and SMA line and color it
        ax1.plot(
            df_chunk.index, df_chunk["EMA"], color="blue", label="EMA", linewidth=1.3
        )
        ax1.plot(
            df_chunk.index, df_chunk["SMA"], color="yellow", label="SMA", linewidth=1.2
        )
        ax1.legend()

        # Plot RSI
        ax2.plot(df_chunk.index, df_chunk["RSI"], color="k", linewidth=2)
        ax2.set_ylabel("RSI")
        ax2.set_ylim(20, 80)
        ax2.grid(True)

        # Plot portfolio value
        ax3.plot(df_chunk.index, self._portfolio_values, color="b", linewidth=2)
        ax3.set_ylabel("Portfolio Value")
        ax3.grid(True)

        # Plot reward and profit
        ax4.plot(df_chunk.index, self._total_reward_history, color="c", linewidth=2)
        ax4.set_ylabel("Total Reward")
        ax4.grid(True)

        # Add current position marker
        current_position = self.df.iloc[self._current_tick]["Close"]
        ax1.scatter(
            self.df.index[self._current_tick],
            current_position,
            marker="*",
            s=200,
            color="purple",
            zorder=10,
        )

        plt.xticks(
            df_chunk.index,
            pd.to_datetime(df_chunk.index).strftime("%Y-%m-%d"),
            rotation=45,
        )
        fig.tight_layout()
        # Adjust all fig spacing to 0
        fig.subplots_adjust(hspace=0, wspace=0)
        # Add the plot to the env list
        self._plots.append(plt.gcf())

    def _export_plot2vid(self):
        """Export the plots to a video"""
        if not self._plots:
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
            "trading_gym/data/animation.mp4",
            dpi=100,
        ):
            # iterate over the list of plots and add them to the animation
            for p in self._plots:
                plt.clf()
                plt.gca().imshow(plot2image(p))
                writer.grab_frame()

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

        actions.append((Action.HOLD, 0))  # Hold action
        return actions

    @staticmethod
    def action_to_string(action_number: Literal[0, 1, 2]):
        """Convert an action number to a string"""
        return str(Action._value2member_map_[action_number])
