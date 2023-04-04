from __future__ import annotations

import sqlite3 as sql
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..type import Action

if TYPE_CHECKING:
    from datetime import datetime


class OrderHandler:
    """Order handler to keep track of past orders"""

    def __init__(self):
        self.conn: sql.Connection | None = None
        self._portfolio_orders: deque[str] = deque()
        self.latest_order: tuple[datetime, int, float, int, float] | None = None
        self._latest_profit: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        )  # (profit, sell_tax, buy_tax)
        self._init_db()

    def _init_db(self):
        # Create a connection pool with 5 connections
        self.conn = sql.connect(
            "trading_gym/data/orders.db",
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
        return len(self._portfolio_orders)

    @property
    def latest_profit(self) -> float:
        """Get the latest profit"""
        # Profit - Sell tax - Buy tax
        return self._latest_profit[0] - self._latest_profit[1] - self._latest_profit[2]

    def add(self, action: Action, price: float, date: datetime) -> tuple[float, float]:
        """Add an order
        Parameters
        ----------
        action : Action
            The action to take
        price : float
            The price of the order
        date : datetime
            The date of the order
        Returns
        -------
            tuple(float, float)
                The total cost and the total tax
        """
        if (
            action != Action.HOLD
            and self.latest_order
            and self.latest_order[1] == action
        ):
            fee = self.calc_tax(price, self.latest_order[3] + 1, action)
            with self.conn:
                date = self.latest_order[0]
                cur = self.conn.cursor()
                cur.execute(
                    """
                    UPDATE orders
                    SET quantity = quantity + 1, trade_fee = ?
                    WHERE date = ?;""",
                    (fee, date.date().isoformat()),
                )
                self.latest_order = (
                    date,
                    action,
                    price,
                    self.latest_order[3] + 1,
                    fee,
                )
        else:
            if self.latest_order and self.latest_order[0] == date:
                return 0.0, 0.0
            fee = self.calc_tax(price, 1, action) if action != Action.HOLD else 0.0
            qtn = 1 if action != Action.HOLD else 0
            with self.conn:
                cur = self.conn.cursor()
                cur.execute(
                    """
                    INSERT INTO orders (date, action, price, quantity, trade_fee)
                    VALUES (?, ?, ?, ?, ?)""",
                    (date.date().isoformat(), int(action), price, qtn, fee),
                )
                self.latest_order = date, action, price, qtn, fee

        if action == Action.BUY:
            self._portfolio_orders.append(date.date().isoformat())
        elif action == Action.SELL:
            self.calc_profit()
        if action != Action.SELL and self.latest_order[2] == Action.SELL:
            # Reset the profit and tax if the latest order
            # is a sell order and the current order is not a sell order
            self._latest_profit = 0.0, 0.0, 0.0
        cost_without_fee = (
            price * self.latest_order[3] * (1 if action == Action.BUY else -1)
        )

        return cost_without_fee, fee

    def calc_profit(self) -> None:
        """Get the latest profit"""
        # Sell order information
        date, _, sell_price, _, sell_fee = self.latest_order

        # Get the buy order information
        buy_date = self._portfolio_orders.popleft()
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT price, trade_fee
            FROM orders
            WHERE date = ? AND action = ?;""",
            (buy_date, int(Action.BUY)),
        )
        buy_price, buy_fee = cur.fetchone()
        cur.close()

        # Calculate the profit
        profit = (sell_price - buy_price) + self._latest_profit[0]
        buy_fee = (
            buy_fee
            if self._latest_profit[2] == buy_fee
            else buy_fee + self._latest_profit[2]
        )
        self._latest_profit = profit, sell_fee, buy_fee

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
        self._portfolio_orders.clear()
        self._latest_profit = 0.0, 0.0, 0.0
        self.latest_order = None
