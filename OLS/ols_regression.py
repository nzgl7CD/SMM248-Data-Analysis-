import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class OLS:
    def __init__(self, dataset, model) -> None:
        self._dataset = dataset
        self._model = model
        self.adjust_data(self._dataset)
        self._regression = self.regression(self._model, self._dataset)
        
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
        print(self.get_regression())  # Call the get_regression method
        print('\n')
    
    def adjust_data(self, dataset):
        dataset['LSALARY'] = np.log(dataset['SALARY'])		
        dataset['ROE2'] = dataset['ROE'] ** 2			
        dataset['ROE3'] = dataset['ROE'] ** 3			

    def plot_data(self):
        plt.hist(self._dataset['SALARY'], 30, edgecolor='black', 
                 weights=np.ones_like(self._dataset['SALARY']) / len(self._dataset))
        plt.xlabel('SALARY')
        plt.ylabel('Density')
        plt.title('Histogram')
        plt.show()
    
    def regression(self, model, dataset):
        return sm.OLS.from_formula(model, dataset).fit()  # Fit the OLS model

    def log_regression_result(func):
        """Decorator to log regression results."""
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            print("Logging regression results...")
            print(result.summary())  # Log the summary of the regression
            return result
        return wrapper
    
    @log_regression_result
    def get_regression(self):
        return self._regression

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
    path = 'OLS/CEOunbal.xlsx'  
    CEOdata = pd.read_excel(path)   
    model = 'LSALARY ~ ROE + ROS + SALES' 
    ols_object = OLS(CEOdata, model)
    ols_object.display()
    ols_object.plot_data()  

main()
