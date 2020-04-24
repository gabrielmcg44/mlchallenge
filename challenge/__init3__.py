import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


window = 5
dist_pred = 4


df = pd.read_csv('data/data.csv')

#df.plot(kind='scatter',x='year',y='value',color='red')
#plt.show()

df['ticker'] = df['indicator']+'/'+df['country']
print(df.head())

tickers = df.ticker.unique().tolist()
print(len(tickers))
tickers_id = range(len(tickers))
tickers_dict = {tickers[i]:tickers_id[i] for i in range(len(tickers_id))}
#print(tickers_dict)


#reg = LinearRegression().fit(X[:27,:], y[:27])
#y_test = reg.predict(np.array([[0, 0, 2013],[0, 0, 2014],[0, 0, 2015]]))
#print(y_test)

