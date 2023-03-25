import pandas as pd


class Indicators:
    def __init__(self, data: pd.DataFrame):
        self.close = data["Close"]

    def add_all(self, sma_period: int = 20, ema_period: int = 20, rsi_period: int = 14):
        """Adds all indicators to the data"""
        sma = self.sma(sma_period)
        ema = self.ema(ema_period)
        rsi = self.rsi(rsi_period)

        # Return all the indicators into a single dataframe
        return pd.DataFrame({"SMA": sma, "EMA": ema, "RSI": rsi})

    def sma(self, period: int) -> pd.Series:
        """Adds the simple moving average to the data
        :param period: The period to calculate the moving average
        """
        # Calculate the simple moving average
        return self.close.rolling(window=period).mean()

    def ema(self, period: int) -> pd.Series:
        """Adds the exponential moving average to the data
        :param period: The period to calculate the moving average
        """
        # Calculate the exponential moving average
        return self.close.ewm(span=period, adjust=False, ignore_na=True).mean()

    def rsi(self, period: int) -> pd.Series:
        """Adds the relative strength index to the data
        :param period: The period to calculate the RSI
        """
        # Calculate the difference in price from previous step
        delta = self.close.diff()

        # Get rid of the first row, which is NaN since it did not have a previous
        # row to calculate the differences
        delta = delta[1:]

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        roll_up1 = up.ewm(span=period).mean()
        roll_down1 = down.abs().ewm(span=period).mean()

        # Calculate the RSI based on EWMA
        rs = roll_up1 / roll_down1
        return 100.0 - (100.0 / (1.0 + rs))
