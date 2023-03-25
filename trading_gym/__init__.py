"""
A basic gym environment for trading stocks

:copyright: (c) 2023-present drlove2002
:license: MIT, see LICENSE for more details.
"""

from gymnasium import register

__version__ = "1.0.0"

register(
    id="stocks-v0",
    entry_point="trading_gym.envs:StocksEnv",
    kwargs={
        "window_size": 10,
        "frame_bound": (10, -1)
    }
)