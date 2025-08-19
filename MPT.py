import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimisation

stocks = [ "AAPL","WMT","TSLA"]
NUM_TRADING_DAYS = 252
start_date = '2017-01-01'
end_date = '2025-08-12'
NUM_portfolios = 2000
def download_data():# dictionary of each stock-> and values +then returns it as a table
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start = start_date, end = end_date)['Close'] # how does it iterate all the stocks if no loops
    return pd.DataFrame(stock_data)

def show_data(data):#then prints the graph of the data
    data.plot(figsize=(12,8))
    plt.show()

def calculate_return(data):#then finds the changes per day
    log_return = np.log(data/data.shift(1)) # again how does it multiple times with no loop  and how does i do it for each stock
    return log_return[1:]

def show_statistics(returns):#finds the  annualised mean
    print(returns.mean()* NUM_TRADING_DAYS )
    print(returns.cov()* NUM_TRADING_DAYS )

def show_mean_variance(returns, weights):# returns =means* weights * annulaised
    portfolio_returns = np.sum(returns.mean() * weights)* NUM_TRADING_DAYS
    portfolio_volatility =np.sqrt( np.dot(weights.T,np.dot(returns.cov()*NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean(return):", portfolio_returns)
    print("expected portfolio volatility:" ,portfolio_volatility)

def generate_portfolios(returns):
    portfolio_means =[] # does this just store the generated values
    portfolio_risks= []
    portfolio_weights=[]
    for _ in range(NUM_portfolios):
        w =np.random.random(len(stocks))
        w /= np.sum(w)# how are the weights always 1 total where is the condition for this
        portfolio_weights.append(w) # what does this do
        portfolio_means.append(np.sum(returns.mean()*w)*NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T,np.dot(returns.cov()*NUM_TRADING_DAYS, w))))
    return (np.array(portfolio_weights),np.array(portfolio_means),np.array(portfolio_risks))

def show_portfolios(portfolio_returns, volatilities):

    # convert to numpy arrays
    rets = np.asarray(portfolio_returns)
    vols = np.asarray(volatilities)

    # avoid division by zero in color computation
    color = np.divide(rets, vols, out=np.zeros_like(rets), where=vols!=0)
    plt.figure(figsize=(12, 8))
    plt.scatter(vols, rets, c=color, marker='o')
    plt.grid(True)
    plt.xlabel("Expected volatility (annualized)")
    plt.ylabel("Expected return (annualized)")
    plt.colorbar(label="Return / Volatility (proxy Sharpe)")
    plt.title("Random portfolios: return vs volatility")
    plt.show()

def statistics( weights,returns):
    portfolio_return = np.sum(returns.mean()* weights)*NUM_TRADING_DAYS
    portfolio_volatility =np.sqrt(np.dot(weights.T , np.dot(returns.cov()*NUM_TRADING_DAYS,weights)))
    return np.array([portfolio_return, portfolio_volatility,portfolio_return/portfolio_volatility])

def min_function_sharpe(weights,returns):
    return -statistics(weights,returns)[2] # for sharpe

def optimize_portfolio(weights,returns):
    constraints= {'type':'eq','fun':lambda x:np.sum(x)-1}
    bounds= tuple( (0,1)for _ in range(len(stocks))) # what does tuple do
    x0= weights[0]# what is weights[0] for?
    return optimisation.minimize(fun=min_function_sharpe, x0=x0,args=(returns,),method= 'SLSQP',bounds = bounds, constraints= constraints)
# why does there need to be a comma after returns
def print_optimal_portfolio(optimum,returns):
    print("optmimal portfolio:",optimum['x'].round(3)) # what is round 3
    print("expected return volatility and sharpe ratio",statistics(optimum['x'].round(3),returns))

def show_optimal_portfolio_return(opt,rets,portfolio_rets,portfolio_vols):
    plt.figure(figsize=(10,6))
    plt.scatter(portfolio_vols,portfolio_rets,c=portfolio_rets/portfolio_vols,marker='o')
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected return (annualized)")
    plt.colorbar(label="Return / Volatility (proxy Sharpe)")
    plt.plot(statistics(opt['x'],rets)[1],statistics(opt['x'],rets)[0],'g*',markersize=20.0)
    plt.show()

if __name__ == "__main__":
    data_set = download_data()#stores the ticker in a dictionary per stock
    show_data(data_set)#plots the data
    log_daily_returns = calculate_return(data_set) #finds the set of returns over time
    show_statistics(log_daily_returns)#does the annualised mean and variance of the cov the panda means that it can be printed

    weights, means, risks = generate_portfolios(log_daily_returns)# makes arrays of weights means risks using different weights for the set of stocks
    show_portfolios(means, risks)#puts in returns and risks in from above and prints graph
    optimum = optimize_portfolio(weights,log_daily_returns)#returns the sharpe closest to 1?
    print_optimal_portfolio(optimum,log_daily_returns)
    show_optimal_portfolio_return(optimum,log_daily_returns,means,risks)

