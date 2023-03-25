"""
A basic gym environment for trading stocks

:copyright: (c) 2023-present drlove2002
:license: MIT, see LICENSE for more details.
"""

from copy import deepcopy

from gymnasium import register

from trading_gym.data import RELIANCE

__version__ = "1.0.0"

register(
    id="stocks-v0",
    entry_point="trading_gym.envs:StocksEnv",
    kwargs={"df": deepcopy(RELIANCE), "window_size": 10},
)
