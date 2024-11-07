import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from scipy.stats import norm


class Stocks:
    def __init__(self,data):
        self.df=data
        self.fix_data()
        self.add_technical_analysis()
    
    def fix_data(self):
        # self.df['Close']=self.df['Close'].interpolate(method='linear', inplace=True)
        self.df['Log_Return'] = (self.df['Close'] / self.df['Close'].shift(1)).apply(lambda x: np.log(x))
        # print(self.df['Close'])
    
    def add_technical_analysis(self):
        self.df['20_day_ma']=self.df['Close'].rolling(window=20).mean()
        self.df['20_day_vol']=self.df['Close'].rolling(window=20).std()
        self.df['ann_vol']=self.df['Close'].rolling(window=252).std()
        self.df['26_ema']=self.df['Close'].ewm(span=26,adjust=False).mean()
        self.df['12_ema']=self.df['Close'].ewm(span=12,adjust=False).mean()
        self.df['MACD']=self.df['12_ema']-self.df['26_ema']
    
    def get_dataframe(self):
        return self.df

    
def get_stock_data(ticker:str, start_date,end_date):
    data_collected=yf.Ticker(ticker)
    return data_collected.history(start=start_date,end=end_date)
    

start_date=datetime.datetime(2019,1,30)
end_date=datetime.datetime(2024,1,30)
data_input=pd.DataFrame(get_stock_data('META', start_date, end_date))

object=Stocks(data_input)
adjusted_dataframe=object.get_dataframe()



def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


K = 100
r = 0.1
T = 1
sigma = 0.3
S = np.arange(60,140,0.1)

calls = [BS_CALL(s, K, T, r, sigma) for s in S]
puts = [BS_PUT(s, K, T, r, sigma) for s in S]

print(calls)



