# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:21:29 2019

@author: NSinalkar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Position_Salaries.csv")
#this way [:,1:2] , x turns out to be a matrix
 
x=dataset.iloc[:,1:2].values

#y stays a vextor
y=dataset.iloc[:,2].values


from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


plt.scatter(x,y,color='blue')
plt.scatter(x,lin_reg.predict(x),color='red')
plt.scatter(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='green')
plt.show()



plt.scatter(x,y,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()


