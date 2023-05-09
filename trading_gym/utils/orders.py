from __future__ import annotations

import sqlite3 as sql
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..type import Action

if TYPE_CHECKING:
    from datetime import datetime


class OrderHandler:
    """Order handler to keep track of past orders"""

    def __init__(self):
        self._quantity = 0
        self.conn: sql.Connection | None = None
        self._init_db()

    def _init_db(self):
        # Create a connection pool with 5 connections
        self.conn = sql.connect(
            ":memory:",
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )

        cur = self.conn.cursor()
        # Check if the table exists or not
        cur.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='orders';
            """
        )
        res = cur.fetchone()
        if res is None:
            # Create the table
            cur.executescript(
                """
                CREATE TABLE orders
                (date TEXT PRIMARY KEY,
                action INTEGER NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                trade_fee REAL NOT NULL);
                ----------------------------
                CREATE INDEX idx_orders_action ON orders(action);
                """
            )
        cur.close()

    @property
    def positions(self) -> int:
        """Get the number of positions"""
        return self._quantity

    def add(
        self, action: Action, quantity: int, price: float, date: datetime
    ) -> tuple[float, float]:
        """Add an order
        Parameters
        ----------
        action : Action
            The action to take
        quantity : int
            The quantity of the order
        price : float
            The price of the order
        date : datetime
            The date of the order
        Returns
        -------
            tuple(float, float)
                The total cost and the total tax
        """
        fee = self.calc_tax(price, 1, action) if action != Action.HOLD else 0.0
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (date, action, price, quantity, trade_fee)
                VALUES (?, ?, ?, ?, ?)""",
                (date.date().isoformat(), int(action), price, quantity, fee),
            )

        if action == Action.BUY:
            self._quantity += quantity
        elif action == Action.SELL:
            self._quantity -= quantity
        cost_without_fee = price * quantity * (1 if action == Action.BUY else -1)

        return cost_without_fee, fee

    @staticmethod
    def calc_tax(del_price: float, del_qty: int, action: Action) -> float:
        """Calculate delivery charges
        Parameters
        ----------
        del_price : float
            The price of the order
        del_qty : int
            The quantity of the order
        action : Action
            The action to take
        Returns
        -------
            float
                The delivery charges
        """
        price: float = round(del_price, 2)
        qty: float = round(del_qty, 2)

        if (qty == 0) or (price == 0):
            return 0.0

        turnover: float = round(price * qty, 2)
        stt_total: float = round(turnover * 0.001, 2)
        exc_trans_charge: float = round(0.0000345 * turnover, 2)
        dp: float = 15.93 if action == Action.SELL else 0.0
        stax: float = round(0.18 * exc_trans_charge, 2)
        sebi_charges: float = round(
            turnover * 0.000001 + (turnover * 0.000001 * 0.18), 2
        )
        stamp_charges: float = (
            round(turnover * qty * 0.00015, 2) if action == Action.BUY else 0.0
        )
        total_tax: float = round(
            stt_total + exc_trans_charge + dp + stax + sebi_charges + stamp_charges, 2
        )

        return total_tax

    def get(self, action: int, df: pd.DataFrame) -> np.ndarray:
        """Get the orders"""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT date FROM orders
            WHERE action = ? AND date BETWEEN ? AND ?
            """,
            (
                action,
                df.index.min().date().isoformat(),
                df.index.max().date().isoformat(),
            ),
        )
        res = cur.fetchall()
        cur.close()
        return pd.to_datetime(np.array(res).flatten()).to_numpy()

    def reset(self):
        """Reset the orders"""
        with self.conn:
            self.conn.cursor().execute("DELETE FROM orders WHERE 1")
