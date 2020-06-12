import numpy as np
import pandas as pd
import pickle


df = pd.read_csv('hiring.csv')
df['experience'].fillna(0,inplace = True)


df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace = True)

x = df.iloc[:,:3]

def convert(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
                 'eight':8,'nine':9,'ten':10,'zero':0,0:0,'eleven':11}
    return word_dict[word]
x['experience'] = x['experience'].apply(lambda x : convert(x))

y = df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)


pickle.dump(reg, open('model.pkl','wb'))

