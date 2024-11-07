import pandas as pd						
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
import statsmodels.stats.diagnostic as smdiag
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.stattools as stats
import statsmodels.tsa.ar_model as sar


CEOdata=pd.read_excel('OLS\CEOunbal.xlsx')
CEOdata['LSALARY']=np.log(CEOdata['SALARY'])
CEOdata['LSALES']=np.log(CEOdata['SALES'])
CEOdata['ROExSALES']=CEOdata['ROE']*CEOdata['SALES']

model='SALARY~ROE+ROS+SALES'

regression1=smf.ols(model, CEOdata).fit()
CEOdata=CEOdata.dropna()


x=CEOdata[['ROE', 'ROS', 'SALES', 'ROExSALES']]
print("Correlation Matrix")
print(x.corr())
for i in range(3): print()
vif_values=pd.DataFrame() #create empty DataFrame to store results
vif_values["Regressors"]=x.columns #first variable "regressors" is filled with labels of X variables
vif_values["VIF"] = [oi.variance_inflation_factor(x.values, i) for i in range(len(x.columns))] #VIFs
print(vif_values) 

# Heteroskedecity
residuals=regression1.resid
predicted=regression1.fittedvalues
plt.xlabel('Predicted values of RD')
plt.ylabel('Residuals')
plt.scatter(predicted, residuals)
plt.show()

print('\n'*3)

# White (1980 test)

white_test = smdiag.het_white(regression1.resid,  regression1.model.exog)
print("White (1980) LM test for heteroskedasticity:")
print("LM test", white_test[0])
print("p-value", white_test[1])
print("F test", white_test[2])
print("p-value", white_test[3])

# Fix Hetsked

regression1=smf.ols(model,CEOdata).fit(cov_type='HC1')
print(regression1.summary())


