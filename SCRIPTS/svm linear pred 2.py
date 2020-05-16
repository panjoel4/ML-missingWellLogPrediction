# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:23:14 2020

@author: ASUS
"""


"""
heatmap (correlation matrix)
polinomial and radial
using trained data in 2 well
compare the mean square error between previous and new model
plot ditimpa
""" 
import pandas as pd
import matplotlib.pyplot as plt
import lasio
import seaborn as sns
las = lasio.read(r'puk1.las')

"""
depth in ft
DT in mus/ft
rhob in g/cc
Vp in m/s
P Impedance in m/s g/cc
"""

"""
load sumur 1
"""

df=las.df()
gr_max=df['GR'].max()
gr_min=df['GR'].min()
df['VSH']=(df['GR']-gr_min)/(gr_max-gr_min)
df['Vp']=1/df['DT']*10**(6)
df['P_imp']=df['Vp']*df['RHOB']

#rename column
df=df.rename(columns={'NPHI_LS':'NPHI'})

#drop column which have no record
df_drop=df.dropna(subset=['P_imp'],axis=0)

#select only at the target area
df_target=df_drop.loc[3337.79:3751.5].reset_index()

"""
load sumur 2
"""
las2 = lasio.read(r'pum1.las')

df2=las2.df()
gr_max=df2['GR'].max()
gr_min=df2['GR'].min()
df2['VSH']=(df2['GR']-gr_min)/(gr_max-gr_min)
df2['Vp']=1/df2['DT']*10**(6)
df2['P_imp']=df2['Vp']*df2['RHOB']

#rename column
df2=df2.rename(columns={'NPHI_LS':'NPHI'})

#drop column which have no record
df2_drop=df2.dropna(subset=['P_imp'],axis=0)

#select only at the target area
df2_target=df2_drop.loc[3327.15:3701.74].reset_index()

#combine sumur 1 dan 2
df_com=pd.concat([df_target,df2_target]).reset_index(drop=True)
df_com.set_index('DEPTH',inplace=True)
df_com.drop(columns=['GR','DT'],inplace=True)

corrmap=df_com.corr()
sns.heatmap(corrmap,annot=True)
plt.show()

# split X and y
X=df_com[['ILD','VSH','Vp','P_imp']]
y=df_com[['NPHI']]

#test and train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=0)

#standardizing data
from sklearn.preprocessing import StandardScaler

sc=StandardScaler().fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

#SVM regression model
from sklearn.svm import SVR
svr=SVR(kernel='linear',C=1)
svr.fit(X_train_std,y_train) 

#predict value using trained model
y_test_pred=svr.predict(X_test_std)
y_train_pred=svr.predict(X_train_std)  

#model performance
from sklearn.metrics import mean_squared_error,r2_score
error=mean_squared_error(y_test,y_test_pred) 
train_r2_score=r2_score(y_train, y_train_pred)
test_r2_score=r2_score(y_test, y_test_pred)
print("r2 score of well 1 & 2 is: "+str(test_r2_score))


"""
Load sumur 3
"""

las3 = lasio.read(r'pux2.las')


df3=las3.df()
gr_max=df3['GR'].max()
gr_min=df3['GR'].min()
df3['VSH']=(df3['GR']-gr_min)/(gr_max-gr_min)
df3['Vp']=1/df3['DT']*10**(6)
df3['P_imp']=df3['Vp']*df3['RHOB']

#rename column
df3=df3.rename(columns={'NPHI_LS':'NPHI'})

#drop column which have no record
df3_drop=df3.dropna(subset=['P_imp'],axis=0)

#select only at the target area
#df3_target=df3_drop.loc[3316.28:3651.67].reset_index() #uxa-1
df3_target=df3_drop.loc[3299.43:3683.01].reset_index() #ux-2

X3=df3_target[['ILD','VSH','Vp','P_imp']]
y3=df3_target[['NPHI']]


sc=StandardScaler().fit(X3)
X3_std=sc.transform(X3)
y3_pred=svr.predict(X3_std)

rmse3=mean_squared_error(y3,y3_pred)
test3_r2_score=r2_score(y3, y3_pred)
print("RMS error for well #3 is: " + str(rmse3))
print("r2 score of well 3 is: "+ str(test3_r2_score))

#plotting check overlay
y3['NPHI_pred']=y3_pred
y3=y3.sort_index()

plt.figure()
plt.scatter(y3['NPHI'],df3_target['DEPTH'],c='k',label='actual',s=10)
plt.scatter(y3['NPHI_pred'],df3_target['DEPTH'],c='r',label='prediction',s=10)
plt.legend(loc='upper left');
plt.ylim(df3_target['DEPTH'].max(),df3_target['DEPTH'].min())
plt.xlim(y3['NPHI'].max(),y3['NPHI'].min())
plt.xlabel('Porosity')
plt.ylabel('Depth')
plt.title('New Model')
plt.show()








"""
#Percobaan dengan hyperparameter polinomial
"""
from sklearn.model_selection import GridSearchCV
params = {'C':(0.05,0.1,0.5,1,5,10,50,100,500),'degree':(1,2,3)}
svrp=SVR(kernel='poly')
svrp_grid_poly=GridSearchCV(svrp,params,scoring='r2',verbose=1)
svrp_grid_poly.fit(X_train_std,y_train) 
poly_best=svrp_grid_poly.best_params_
y_test_poly=svrp_grid_poly.best_estimator_
y_test_pred_poly=y_test_poly.predict(X_test_std)
test_poly_r2_score=r2_score(y_test,y_test_pred_poly)
print("hasil test polynomial untuk sumur 1&2 adalah: " + str(test_poly_r2_score))



#percobaan dengan hyperparameter radial
from sklearn.model_selection import GridSearchCV
params = {'C':(0.05,0.1,0.5,1,5,10,50,100,500),'gamma':(0,0.00001)}
svrp=SVR(kernel='rbf')
svrp_grid_rad=GridSearchCV(svrp,params,scoring='r2',verbose=1)
svrp_grid_rad.fit(X_train_std,y_train) 
rad_best=svrp_grid_rad.best_params_
y_test_rad=svrp_grid_rad.best_estimator_
y_test_pred_rad=y_test_rad.predict(X_test_std)
test_rad_r2_score=r2_score(y_test,y_test_pred_rad)
print("hasil test radial untuk sumur 1&2 adalah: " + str(test_rad_r2_score))





