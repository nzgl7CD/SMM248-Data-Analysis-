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

class Options:
    def __init__(self,dataframe) -> None:
        self.df=dataframe
        self.add_option_premiums()
    
    def add_option_premiums(self):
        '''
        Using Black-Scholes
        TODO: fix teh calculation by splitting it in pieces
        '''
        rf = 0.05  # Risk-free rate

        # Define strikes
        self.df['Call_Strike'] = self.df['Close'] + 5  # Strike 5 above spot 
        self.df['Put_Strike'] = self.df['Close'] - 5  # Strike 5 below spot 

        for idx in range(len(self.df)):
            # Check if any required value is None (or NaN)
            if pd.isnull(self.df['Close'].iloc[idx]) or pd.isnull(self.df['Call_Strike'].iloc[idx]) or pd.isnull(self.df['Put_Strike'].iloc[idx]) or pd.isnull(self.df['ann_vol'].iloc[idx]):
                continue  # Skip to the next iteration if any value is None

            # Calculate d1 and d2 for call option
            d1_call = (np.log(self.df['Close'].iloc[idx] / self.df['Call_Strike'].iloc[idx]) + 
                    (rf + 0.5 * self.df['ann_vol'].iloc[idx]**2) * (30 / 252)) / \
                    (self.df['ann_vol'].iloc[idx] * np.sqrt(30 / 252))

            d2_call = d1_call - self.df['ann_vol'].iloc[idx] * np.sqrt(30 / 252)
            norm_d1_call = norm.cdf(d1_call)
            norm_d2_call = norm.cdf(d2_call)
            

            # Calculate Call Premium
            self.df.at[idx, 'Call_Premium'] = (self.df['Close'].iloc[idx] * norm_d1_call -
                                                self.df['Call_Strike'].iloc[idx] * np.exp(-rf * (30 / 252)) * norm_d2_call)

            # Calculate d1 and d2 for put option
            d1_put = (np.log(self.df['Close'].iloc[idx] / self.df['Put_Strike'].iloc[idx]) + 
                    (rf + 0.5 * self.df['ann_vol'].iloc[idx]**2) * (30 / 252)) / \
                    (self.df['ann_vol'].iloc[idx] * np.sqrt(30 / 252))

            d2_put = d1_put - self.df['ann_vol'].iloc[idx] * np.sqrt(30 / 252)
            norm_d1_put = norm.cdf(-d1_put)  # Corrected sign
            norm_d2_put = norm.cdf(-d2_put)  # Corrected sign

            # Calculate Put Premium
            self.df.at[idx, 'Put_Premium'] = (self.df['Put_Strike'].iloc[idx] * np.exp(-rf * (30 / 252)) * norm_d2_put - 
                                                self.df['Close'].iloc[idx] * norm_d1_put)
        # print(self.df[['Call_Premium']])

    
    
def get_stock_data(ticker:str, start_date,end_date):
    data_collected=yf.Ticker(ticker)
    return data_collected.history(start=start_date,end=end_date)
    

start_date=datetime.datetime(2019,1,30)
end_date=datetime.datetime(2024,1,30)
data_input=pd.DataFrame(get_stock_data('META', start_date, end_date))

object=Stocks(data_input)
adjusted_dataframe=object.get_dataframe()

option_object=Options(adjusted_dataframe)




