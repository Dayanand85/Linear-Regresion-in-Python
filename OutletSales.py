# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:12:30 2022

@author: Dayanand

"""
# loading libraries

import os
import numpy as np
import seaborn as sns
import pandas as pd

# fixing display width

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# setting directory

os.getcwd()
os.chdir("C:/Users/tk/Desktop/DataScience/Analytics Vidhya/DataSet")

# loading files

rawData=pd.read_csv("train_file.csv")
rawData.head()
rawData.shape
predictionData=pd.read_csv("test_file.csv")
predictionData.head()
predictionData.shape

rawData.columns 
predictionData.columns

predictionData["Item_Outlet_Sales"]= 0
predictionData.shape


# Dividing the rawData sets into train & test

from sklearn.model_selection import train_test_split

trainDf,testDf=train_test_split(rawData,train_size=0.8,random_state=2410)
trainDf.shape
testDf.shape

# let us create source column in train,test & prediction datasets

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionData["Source"]="Prediction"

# concat all three datasets

fullDf=pd.concat([trainDf,testDf,predictionData],axis=0)
fullDf.shape

# let us remove identifier columns

fullDf.drop(["Item_Identifier"],axis=1,inplace=True)
fullDf.drop(["Outlet_Identifier"],axis=1,inplace=True)
fullDf.shape

# Check NUll /missing values(Univariate Analysis)

fullDf.isna().sum()

# Univariate Analysis-Missing values
# we have got one column Item_Weight which has missing values which is
# continuous variable

fullDf.dtypes
tempMedian=fullDf.loc[fullDf["Source"]=="Train","Item_Weight"].median()
tempMedian
fullDf["Item_Weight"].fillna(tempMedian,inplace=True)
fullDf.isna().sum()

# We have one column Outlet_Size which is categorical column which
# has missing values

fullDf.dtypes
tempMode=fullDf.loc[fullDf["Source"]=="Train","Outlet_Size"].mode()[0]
tempMode
fullDf["Outlet_Size"].fillna(tempMode,inplace=True)
fullDf.isna().sum()

# Bivariate Analysis-for continuous variables(correlation & scatter plot)

corrDf= fullDf[fullDf["Source"]=="Train"].corr()
corrDf.head()
sns.heatmap(corrDf,xticklabels=corrDf.columns,yticklabels=corrDf.columns,cmap='YlOrBr')

# Bivariate Analysis for categorical variables using boxplot
fullDf.dtypes
categVars=fullDf.columns[fullDf.dtypes==object]
categVars
from matplotlib.pyplot import figure
for colName in categVars:
    figure()
    sns.boxplot(y=fullDf["Item_Outlet_Sales"],x=fullDf[colName])

# Outlier Detection & Correction

fullDf.dtypes
VarForOutlier=["Item_Weight","Item_Visibility","Item_MRP"]

VarBeforeOutlierCorrection=fullDf[VarForOutlier].describe()

for colName in VarForOutlier:
    Q1=np.percentile(fullDf.loc[fullDf["Source"]=="Train",colName],25)
    Q3=np.percentile(fullDf.loc[fullDf["Source"]=="Train",colName],75)
    IQR=Q3-Q1
    LB=Q1-1.5*IQR
    UB=Q3+1.5*IQR
    fullDf[colName]=np.where(fullDf[colName]<LB,LB,fullDf[colName])
    fullDf[colName]=np.where(fullDf[colName]>UB,UB,fullDf[colName])

VarAfterOutlierCorrection=fullDf[VarForOutlier].describe()

# Dummy variable creation
fullDf2=pd.get_dummies(fullDf,drop_first=True)
fullDf2.shape

# Add intercept column

from statsmodels.api import add_constant

fullDf2=add_constant(fullDf2)
fullDf2.shape

# sampling the data into train,test & prediction

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1).copy()
trainDf.shape
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test"],axis=1).copy()
testDf.shape
predictionDf=fullDf2[(fullDf2["Source_Train"]==0) & (fullDf2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1).copy()
predictionDf.shape

# Dependent & Independent Variable

trainX=trainDf.drop(["Item_Outlet_Sales"],axis=1)
trainY=trainDf["Item_Outlet_Sales"]
testX=testDf.drop(["Item_Outlet_Sales"],axis=1)
testY=testDf["Item_Outlet_Sales"]
predictionX=predictionDf.drop(["Item_Outlet_Sales"],axis=1)
predictionY=predictionDf["Item_Outlet_Sales"]

trainX.shape
trainY.shape
testX.shape
testY.shape
predictionX.shape
predictionY.shape

# VIF check

from statsmodels.stats.outliers_influence import variance_inflation_factor

maxVIF=5
cutoffVIF=5
trainXcopy=trainX.copy()
highVIFcolumn=[]
counter=0

while(maxVIF>=cutoffVIF):
    tempDf=pd.DataFrame()
    
    tempDf["VIF"]=[variance_inflation_factor(trainXcopy.values,i) for i in range (trainXcopy.shape[1])]
    tempDf["Col"]=trainXcopy.columns
    tempDf.dropna(inplace=True)
    
    tempColName=tempDf.sort_values(["VIF"],ascending=False).iloc[0,1]
    maxVIF=tempDf.sort_values(["VIF"],ascending=False).iloc[0,0]
    counter=counter+1
    
    if(maxVIF>cutoffVIF):
        trainXcopy=trainXcopy.drop(tempColName,axis=1)
        highVIFcolumn.append(tempColName)
    counter=counter+1

highVIFcolumn

highVIFcolumn.remove("const")
trainX=trainX.drop(highVIFcolumn,axis=1)
trainX.shape
testX=testX.drop(highVIFcolumn,axis=1)
testX.shape
predictionX=predictionX.drop(highVIFcolumn,axis=1)
predictionX.shape

# Model Building
from statsmodels.api import OLS
Mod_Def=OLS(trainY,trainX)
M1=Mod_Def.fit()
M1.summary()

M1_Pval=M1.pvalues

# Significant Variables selection

from sklearn.ensemble import RandomForestRegressor
RM1=RandomForestRegressor(random_state=2410).fit(trainX,trainY)
ImpVar=pd.DataFrame()
ImpVar["Feature_Val"]=RM1.feature_importances_
ImpVar["Col"]=trainX.columns
Imp_Median=ImpVar["Feature_Val"].median()
tempColName=ImpVar.sort_values(["Feature_Val"],ascending=False).iloc[0,0]

tempVarName=[]
for i in range(ImpVar.shape[0]):
    print(i)
    if ImpVar["Feature_Val"][i]<=Imp_Median:
        tempVarName.append(ImpVar["Col"][i])

tempVarName.remove("const")

trainX=trainX.drop(tempVarName,axis=1)
trainX.shape

testX=testX.drop(tempVarName,axis=1)
testX.shape

predictionX=predictionX.drop(tempVarName,axis=1)
predictionX.shape

# Model Building with significant variables
M2=OLS(trainY,trainX).fit()
M2.summary()

# prediction on test Data
Test_Predict=M2.predict(testX)
Test_Predict[1:6]

# Model Diagnostics plots(validating the assumption)

# Homoskedasticity check
sns.scatterplot(M2.fittedvalues,M2.resid)

# Normality check
sns.distplot(M2.resid)

# Model Evaluation

np.sqrt(np.mean((Test_Predict-testY)**2))

# MAPE
        
(np.mean(np.abs(((testY-Test_Predict)/testY))))*100

# prediction on prediction
rawData.columns
predictionData["Prediction"]=M2.predict(predictionX)
OutputFile=pd.DataFrame()
OutputFile["Item_Identifier"]=predictionData["Item_Identifier"]
OutputFile["Outlet_Identifier"]=predictionData["Outlet_Identifier"]
OutputFile["Prediction"]=predictionData["Prediction"]

OutputFile.to_csv("OutputFile.csv")
