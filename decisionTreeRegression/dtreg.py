#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: Muhammed Didin
"""

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#Data Import
veriler = pd.read_csv('dataset/maaslar.csv')

x = veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]
X=x.values
Y=y.values
#Linear regression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.show()

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y) 

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X)) 