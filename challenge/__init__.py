import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import random
from _internal import evaluate
import pickle


class regression_model(object):

    def __init__(self):
        pass

    def train_my_model(self,data):
        
        df = data
        df['ticker'] = df['indicator']+'/'+df['country']

        x_train = []
        y_train = []

        counter = 0

        tickers = df.ticker.unique().tolist()
        print(len(tickers))
        tickers_id = range(len(tickers))
        self.tickers_dict = {tickers[i]:tickers_id[i] for i in range(len(tickers_id))}

        coefs = []

        for tic in tickers:
            counter+=1
            print(counter, end="\r", flush=True)
            df_tic = df[df['ticker']==tic]
            df_tic = df_tic.reset_index()
            np_indc = df_tic[['year','value']].values
            X = np_indc[:,0].reshape(-1,1)
            y = np_indc[:,1].reshape(-1,1)
            reg = LinearRegression().fit(X,y)
            coefs.append([reg.coef_[0,0],reg.intercept_[0]])

        self.coefs_np = np.asarray(coefs)
        
    
    def predict(self,test):
    
        test['ticker'] = test['indicator']+'/'+test['country']
        test['value'] = test['year']*self.coefs_np[test['ticker'].map(self.tickers_dict),0] + self.coefs_np[test['ticker'].map(self.tickers_dict),1]

        my_answer = test.drop(columns=['ticker'])
        my_answer.to_csv ('my_answer_linear.csv', index = False, header=True)
        self.y_pred = test['value'].values
        return self.y_pred


data = pd.read_csv('data/data.csv')
test = pd.read_csv('data/test.csv')

model = regression_model()
model.train_my_model(data)
y_pred = model.predict(test)

pickle.dump(y_pred,open("y_pred_linear","wb"))

metrics = evaluate.evaluate_regression(y_pred)
print(metrics)

