from  typing import NamedTuple
from  datetime  import  datetime
from .enums import Action

class Order(NamedTuple):
    """Order"""
    action: Action
    price: float
    date: datetime
    quantity: int
    trade_fee: float
