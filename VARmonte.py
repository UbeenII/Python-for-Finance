import pandas as pd
import yfinance as yf
from scipy.stats import norm
import numpy as np
import datetime


def download_data(stock, start_date, end_date):

    ticker = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)

    return pd.DataFrame(ticker['Close'])


class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S  # value of investment at t=0
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, self.iterations)

        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) +self.sigma * np.sqrt(self.n) * rand)
        percentile = np.percentile(stock_price, (1 - self.c) * 100)
        return self.S - percentile


if __name__ == '__main__':
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2025, 8, 12)
    stock_data = download_data('C', start, end)
    stock_data['returns'] = np.log(stock_data['C'] / stock_data['C'].shift(1))
    stock_data = stock_data[1:]
    C = 0.95  # Confidence level
    n = 1  # Time horizon (days)
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])
    model = ValueAtRiskMonteCarlo(1000, mu, sigma, C, n, 100000)
    print('Value at Risk:', model.simulation())