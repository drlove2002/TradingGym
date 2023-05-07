from concurrent import futures
from typing import Literal, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..type import Action
from ..utils import Indicators, OrderHandler, get_logger

logger = get_logger(__name__)
INF = 1e10
WINDOW_SIZE = 24


class StocksEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1, "render_title": "StocksEnv"}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        initial_balance: float = 10_000.0,
        feature_size: int = 8,
        df_scaled: Optional[pd.DataFrame] = None,
    ):
        """
        Stock trading environment
        :param df: DataFrame with stock prices and volume in OHLCV format
        """
        super().__init__()
        self.df = df
        self.df_scaled: pd.DataFrame | None = df_scaled
        self._process_data()
        self._thread_pool = futures.ThreadPoolExecutor(max_workers=4)

        self.action_space = spaces.Discrete(len(Action))  # Buy, Sell, or Hold
        self.observation_space = spaces.Dict(
            {
                "balance": spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float32),
                "equity": spaces.Box(low=-INF, high=INF, shape=(1,), dtype=np.float32),
                "quantity": spaces.Discrete(2),
                "features": spaces.Box(
                    low=0,
                    high=1,
                    shape=(feature_size * (WINDOW_SIZE + 1),),
                    dtype=np.float64,
                ),
                "future_price": spaces.Box(
                    low=0, high=1, shape=(WINDOW_SIZE,), dtype=np.float64
                ),
            }
        )

        # episode
        self._episode = 0
        self.tick = 0
        self._plots: list[plt.Figure] = []
        self._orders = OrderHandler()
        self._init_balance = initial_balance
        self._balance = self._last_balance = self._init_balance
        self._start_tick = WINDOW_SIZE
        self._end_tick = len(self.df) - WINDOW_SIZE
        self._done = False
        self._current_tick = self._start_tick
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
        if "RSI" not in self.df.columns:
            ind = Indicators(self.df).add_all()
            self.df = self.df.join(ind).dropna()

        if self.df_scaled is None:
            # scale the data using min max scaler
            self.df_scaled = self.df.copy()
            for col in self.df.columns:
                col_min = self.df_scaled[col].min()
                col_max = self.df_scaled[col].max()
                self.df_scaled[col] = (self.df_scaled[col] - col_min) / (
                    col_max - col_min
                )

    @property
    def _qtn(self) -> int:
        """Get the quantity of shares"""
        return self._orders.positions

    @property
    def _equity(self) -> float:
        """Get the current equity. The unrealized profit"""
        if self._qtn == 0 or self._done:
            return 0.0
        tax = self._orders.calc_tax(self._current_price, self._qtn, Action.SELL)
        return (self._qtn * self._current_price) - tax

    @property
    def _obs(self):
        """Get the observation"""
        features = (
            self.df_scaled.iloc[
                self._current_tick - WINDOW_SIZE : self._current_tick + 1
            ]
            .to_numpy()
            .flatten()
        )
        future_price = (
            self.df_scaled["Close"]
            .iloc[self._current_tick + 1 : self._current_tick + WINDOW_SIZE + 1]
            .to_numpy()
        )
        return {
            "balance": np.array([self._balance], dtype=np.float32),
            "equity": np.array([self._equity], dtype=np.float32),
            "quantity": self._qtn,
            "features": features,
            "future_price": future_price,
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
        # Keep track of the history of portfolio values
        current_value = self._balance + self._equity
        self._portfolio_values[self._current_tick] = current_value

        reward = 0.0

        if action == Action.SELL:
            profit = self._orders.latest_profit
            reward += profit
            if profit > 0:
                # Reward the agent for selling at a profit
                reward *= 5

        if action == Action.BUY:
            reward -= fee

        if action == Action.HOLD:
            reward -= (0.1 * (self._init_balance - current_value)) + self.tick

        return reward

    def step(self, action):
        """Take a step in the environment"""
        if isinstance(action, Action):
            action = action.value

        self.tick += 1

        # We are done if we blow up our balance by 50% or if we reach the end of the data
        if (self._balance <= (self._init_balance * 0.5)) and (self._equity <= 0):
            self._done = True

        if action not in self.legal_actions():
            return self._obs, -self._init_balance, False, self._done, {}

        last_action = (
            self._last_action
        )  # Get the last action before we update the orders
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

        reward = self._get_reward(action, fee)
        observation = self._obs

        if self._current_tick < self._end_tick and action == Action.HOLD:
            # Move to the next tick
            self._current_tick += 1

        return (
            observation,
            reward,
            (self._current_tick >= self._end_tick),
            self._done,
            {},
        )

    def reset(self, seed=None, options=None):
        """Reset the environment data and state"""
        super().reset(seed=seed, options=options)

        self._episode += 1
        self._balance = self._last_balance = self._init_balance
        self._done = False
        self.tick = 0
        self._current_tick = self._start_tick
        self._portfolio_values[:] = self._balance
        self._plots.clear()
        self._orders.reset()

        observation = self._obs
        return observation, {}

    def render(self, mode="human"):
        """Render the stock chart with the current position"""
        if mode != "human":
            return

        if not self._done:
            df = self.df.iloc[: self._current_tick]
            portfolio = self._portfolio_values[: self._current_tick]
        else:
            df = self.df
            portfolio = self._portfolio_values
        chunk_size = len(df) // 180
        chunks = [
            df.iloc[i : i + len(df) // chunk_size]
            for i in range(0, len(df), len(df) // chunk_size)
        ]
        fut = []
        for chunk in chunks:
            start_row = chunk.index[0]
            end_row = chunk.index[-1]
            start_idx = df.index.get_loc(start_row)
            end_idx = df.index.get_loc(end_row)
            # Return when chunk is too small
            if end_idx - start_idx < 10:
                continue
            portfolio_values_chunk = portfolio[start_idx : end_idx + 1]
            fut.append(
                self._thread_pool.submit(self._draw_plot, chunk, portfolio_values_chunk)
            )

        # Wait for all plots to be drawn
        for fut in fut:
            fut.result()

        # Render the plot
        for p in self._plots:
            p.show(warn=False)

    def _draw_plot(self, df: pd.DataFrame, portfolio_chunk: np.ndarray):
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
            mpf.make_addplot([30] * len(df), panel=2, color="r", type="line"),
            mpf.make_addplot([70] * len(df), panel=2, color="r", type="line"),
        ]

        # Mark buy and sell orders in the candlestick chart
        buy_orders = self._orders.get(int(Action.BUY), df)
        sell_orders = self._orders.get(int(Action.SELL), df)
        # get high value for buy orders and low value for sell orders and nan if no order
        if buy_orders.shape[0]:
            # Get those buy orders which are not in the current chunk
            buy_orders = df.loc[buy_orders, "Low"].reindex(df.index).values * 0.99
            plot_data.append(
                mpf.make_addplot(
                    buy_orders,
                    type="scatter",
                    marker="^",
                    markersize=100,
                    color="green",
                    panel=0,
                    alpha=0.5,
                )
            )
        if sell_orders.shape[0]:
            sell_orders = df.loc[sell_orders, "High"].reindex(df.index).values * 1.01
            plot_data.append(
                mpf.make_addplot(
                    sell_orders,
                    type="scatter",
                    marker="v",
                    markersize=100,
                    color="red",
                    panel=0,
                    alpha=0.5,
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
            panel_ratios=(4, 1, 1, 1),
            figsize=(10, 9),
        )

        self._plots.append(fig)
        logger.info("Plot drawn for episode %s", self._episode)

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
            and self._balance // self._current_price > 0
            and self._qtn == 0
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
