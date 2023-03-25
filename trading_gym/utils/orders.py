from __future__ import annotations

import sqlite3 as sql
from typing import TYPE_CHECKING

from ..type import Action

if TYPE_CHECKING:
    from datetime import datetime


class OrderHandler:
    """Order handler to keep track of past orders"""

    def __init__(self):
        self.conn = sql.connect("trading_gym/data/orders.db")
        self.positions = 0
        self._latest_sell_date: str = ""
        self._init_db()

    def _init_db(self):
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
            cur.execute(
                """
                CREATE TABLE orders
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                action INTEGER NOT NULL,
                price REAL NOT NULL,
                date TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                trade_fee REAL NOT NULL);
            """
            )
            cur.execute("CREATE INDEX idx_orders_date ON orders(date);")
            cur.execute("CREATE INDEX idx_orders_action ON orders(action);")
        cur.close()

    @property
    def latest_profit(self) -> float:
        """Get the latest profit"""
        if not self._latest_sell_date:
            return 0.0
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                SUM(profit) AS total_profit
                FROM (
                    SELECT
                        -- Calculate profit from past buy orders
                        (o_sell.price - o_buy.price) * MIN(o_sell.quantity, o_buy.quantity)
                         - o_buy.trade_fee - o_sell.trade_fee AS profit,
                        -- Count the number of shares sold from this buy order
                        SUM(MIN(o_sell.quantity, o_buy.quantity)) OVER
                        (ORDER BY o_buy.date DESC, o_buy.id DESC) AS sold_quantity
                    FROM
                        (
                            -- Subquery to filter the sell order
                            SELECT *
                            FROM orders INDEXED BY idx_orders_date
                            WHERE date = ?
                        ) AS o_sell
                        INNER JOIN
                        (
                            -- Subquery to filter the buy orders
                            SELECT *
                            FROM orders INDEXED BY idx_orders_action
                            WHERE action = 0
                              AND quantity > 0
                        ) AS o_buy
                        ON o_buy.date < o_sell.date
                    WHERE o_sell.action = 1
                    ORDER BY o_buy.date DESC, o_buy.id DESC
                ) AS profits
                WHERE sold_quantity >= (
                    SELECT quantity
                    FROM orders
                    WHERE date = ?
                );
            """,
            (self._latest_sell_date, self._latest_sell_date),
        )
        res = cur.fetchone()
        cur.close()
        if not res:
            return 0.0
        return res[0]

    def add(
        self, action: Action, price: float, date: datetime, quantity: int = 0
    ) -> tuple[float, float]:
        """Add an order
        Parameters
        ----------
        action : Action
            The action to take
        price : float
            The price of the order
        date : datetime
            The date of the order
        quantity : int, optional
            The quantity of the order, by default 0
        Returns
        -------
            tuple(float, float)
                The total cost and the total tax
        """
        fee = self.calc_tax(price, quantity, action)
        if action == Action.BUY:
            self.positions += quantity
        elif action == Action.SELL:
            self.positions -= quantity
            self._latest_sell_date = date.date().isoformat()
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (action, price, date, quantity, trade_fee)
                VALUES (?, ?, ?, ?, ?)""",
                (int(action), price, date.date().isoformat(), quantity, fee),
            )
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

        stt_total: float = round(price * 0.001, 2)
        exc_trans_charge: float = round(0.0000345 * price, 2)
        dp: float = 15.93 if action == Action.SELL else 0.0
        stax: float = round(0.18 * exc_trans_charge, 2)
        sebi_charges: float = round(price * 0.000001 + (price * 0.000001 * 0.18), 2)
        stamp_charges: float = (
            round(price * qty * 0.00015, 2) if action == Action.BUY else 0.0
        )
        total_tax: float = round(
            stt_total + exc_trans_charge + dp + stax + sebi_charges + stamp_charges, 2
        )

        return total_tax

    def reset(self):
        """Reset the orders"""
        with self.conn:
            self.conn.cursor().execute("DELETE FROM orders")
        self.positions = 0
