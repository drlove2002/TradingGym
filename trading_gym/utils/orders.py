from __future__ import annotations

from typing import List, TYPE_CHECKING
from ..type import Order, Action

if TYPE_CHECKING:
    from datetime import datetime

class OrderHandler:
    """Order handler to keep track of past orders"""
    def __init__(self):
        self._orders: List[Order]  = []
        self.positions = 0

    def add(
            self,
            action: Action,
            price: float,
            date: datetime,
            quantity: int = 0
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
        self._orders.append(Order(action, price, date, quantity, fee))
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
        stamp_charges: float = round(price * qty * 0.00015, 2) if action == Action.BUY else 0.0
        total_tax: float = round(stt_total + exc_trans_charge + dp + stax + sebi_charges + stamp_charges, 2)

        return total_tax

    def reset(self):
        """Reset the orders"""
        self._orders.clear()
        self.positions = 0

    def get(self) -> List[Order]:
        """Get all orders"""
        return self._orders

    def as_df(self):
        """Get all orders as a DataFrame"""
        import pandas as pd
        return pd.DataFrame(self._orders)
