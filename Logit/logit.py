import pandas as pd						
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as smdiag
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.stattools as stats
import sklearn.metrics as skm  ## binary models
import sklearn.model_selection as skms   ##logit evaluation


class Logit:
    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self.adjust_data(self._dataset)
        self._regression = self.regression(self._dataset)
        
    def display(self):
        print('\n')
        print(self._dataset.head())
        print('\n')
        print('-' * 110)
        print('\n')
        print(self._dataset.describe())
        print('\n')
        print(self._dataset.corr())
        print('\n')
        print(self.get_regression())  
        print('\n')
    
   	

    def useful_commands(self):
        regression = self.get_regression()  # Retrieve regression object
        print(regression.summary())  # Get the “entire” regression output
        print(regression.params)  # Get the B1, B2, … estimates
        print(regression.params[0])  # Get the first parameter (intercept)
        print(regression.params[1])  # Get the second parameter (slope)
        print(regression.bse)  # Get the standard errors
        print(regression.bse[0])  # Get the s.e.(B1)
        print(regression.bse[1])  # Get the s.e.(B2)
        print(regression.df_resid)  # Get the df parameter N-k
        print(regression.resid)  # Print the residuals
        print(regression.predict())  # Print the model predictions (points on the line)
        print(regression.fittedvalues)  # Print the fitted values


def main():
    path = 'Logit\creditscore.xlsx'  
    logitData = pd.read_excel(path) 
    

main()