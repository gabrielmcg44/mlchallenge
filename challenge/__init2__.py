import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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
            model_poli = Pipeline([('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression(fit_intercept=False))])
            reg = model_poli.fit(X,y)
            coefs.append(reg.named_steps['linear'].coef_)

        self.coefs_np = np.asarray(coefs).reshape(-1,4)
        
    
    def predict(self,test):
    
        test['ticker'] = test['indicator']+'/'+test['country']

        test['value'] = self.coefs_np[test['ticker'].map(self.tickers_dict),0] \
                    + self.coefs_np[test['ticker'].map(self.tickers_dict),1]*test['year']\
                    + self.coefs_np[test['ticker'].map(self.tickers_dict),2]*(test['year']**2)\
                    + self.coefs_np[test['ticker'].map(self.tickers_dict),3]*(test['year']**3)

        my_answer = test.drop(columns=['ticker'])
        my_answer.to_csv ('my_answer_polinomial.csv', index = False, header=True)
        self.y_pred = test['value'].values
        #print(self.y_pred)
        return self.y_pred


data = pd.read_csv('data/data_reduced.csv')
test = pd.read_csv('data/test_reduced.csv')

model = regression_model()
model.train_my_model(data)
pickle.dump(model,open("model_mlchallenge","wb"))
#model = pickle.load(open("model_mlchallenge","rb"))
y_pred = model.predict(test)

metrics = evaluate.evaluate_regression(y_pred)
print(metrics)

