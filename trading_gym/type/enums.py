from enum import IntEnum


class Action(IntEnum):
    BUY = 1
    SELL = -1
    HOLD = 0

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.capitalize()
