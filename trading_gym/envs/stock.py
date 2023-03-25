from abc import ABC

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ..type import Action


class TradingEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int, frame_bound: int):
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.shape = (window_size, len(df.columns))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(len(Action))

    def _