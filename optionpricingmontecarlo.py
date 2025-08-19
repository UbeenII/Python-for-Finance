import numpy as np
import yfinance as yf
import math
import scipy.stats  # Changed import
from datetime import datetime, timedelta


def run_convergence_test(trials=10):
    deviations = []
    for _ in range(trials):
        mc_price = OptionPricing(current_price, STRIKE_PRICE,
                                 TIME_TO_EXPIRY, RISK_FREE_RATE,
                                 sigma, 100000).call_option()
        deviations.append(abs(bs_call - mc_price) / bs_call)
    return np.mean(deviations) * 100

def download_close(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data['Close']


def annualized_hist_vol(prices, window_days=252):
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma = returns.std() * math.sqrt(252)
    mu = returns.mean() * 252
    return mu, sigma


def bs_price(S, K, T, r, sigma, option='call'):
    if T <= 0:
        return max(0.0, S - K) if option == 'call' else max(0.0, K - S)

    # Ensure all parameters are floats
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option == 'call':
        return S * scipy.stats.norm.cdf(d1) - K * math.exp(-r * T) * scipy.stats.norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * scipy.stats.norm.cdf(-d2) - S * scipy.stats.norm.cdf(-d1)


class OptionPricing:
    def __init__(self, s0, e, t, rf, sigma, iterations):
        self.S0 = s0
        self.E = e
        self.T = t
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option(self):
        rand = np.random.normal(0, 1, self.iterations)
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) +
                                       self.sigma * np.sqrt(self.T) * rand)
        payoffs = np.maximum(stock_price - self.E, 0)
        return np.exp(-self.rf * self.T) * np.mean(payoffs)

    def put_option(self):
        rand = np.random.normal(0, 1, self.iterations)
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma ** 2) +
                                       self.sigma * np.sqrt(self.T) * rand)
        payoffs = np.maximum(self.E - stock_price, 0)
        return np.exp(-self.rf * self.T) * np.mean(payoffs)






if __name__ == '__main__':
    # Configuration parameters
    TICKER = 'AAPL'
    STRIKE_PRICE = 180
    RISK_FREE_RATE = 0.02
    TIME_TO_EXPIRY = 0.25  # 3 months
    MONTE_CARLO_ITERATIONS = 100000

    # Get historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    prices = download_close(TICKER, start_date, end_date)
    current_price = prices.iloc[-1].item() # Convert to float

    # Calculate historical volatility
    mu, sigma = annualized_hist_vol(prices)
    sigma = sigma.item()  # Ensure float

    # Black-Scholes pricing
    bs_call = bs_price(current_price, STRIKE_PRICE, TIME_TO_EXPIRY,
                       RISK_FREE_RATE, sigma, 'call')

    # Monte Carlo pricing
    mc_model = OptionPricing(
        s0=current_price,
        e=STRIKE_PRICE,
        t=TIME_TO_EXPIRY,
        rf=RISK_FREE_RATE,
        sigma=sigma,
        iterations=MONTE_CARLO_ITERATIONS
    )
    mc_call = mc_model.call_option()

    # Calculate difference
    diff = abs(bs_call - mc_call)
    diff_pct = (diff / bs_call) * 100

    # Print results
    print("Current", TICKER, "price" ,  current_price)
    print("Annualized volatility:", sigma)
    print("Annualized return:" ,mu.item())

    print("\nBlack-Scholes Call Price:", bs_call)
    print("Monte Carlo Call Price:",mc_call)
    print("Difference:" ,diff)
    print(f"Mean deviation across 10 runs:",run_convergence_test())
