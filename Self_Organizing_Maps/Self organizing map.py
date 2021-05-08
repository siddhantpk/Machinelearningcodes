import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import bone, colorbar, pcolor, plot, show
dataset= pd.read_csv('Credit_Card_Applications.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
x= sc.fit_transform(x)

from minisom import MiniSom
som= MiniSom(x=10,y=10, input_len=15, sigma=1.0,learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)

bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,j in enumerate(x):
    w=som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=15,
         markeredgewidth=2)
show()
