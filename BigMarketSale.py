# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 19:33:31 2021

@author: Dayanand
"""
# Loading library
import os
import seaborn as sns
import pandas as pd
import numpy as np

# fixing display size

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# Loading file

trainRaw= pd.read_csv("C:\\Users\\tk\\Desktop\\DataScience\\DataSets\\BigMartSales\\train-set.csv")
predictionRw=pd.read_csv("C:\\Users\\tk\\Desktop\\DataScience\\DataSets\\BigMartSales\\test-set.csv")

trainRaw.shape
trainRaw.columns
predictionRw.shape
predictionRw.columns

trainRaw.dtypes
predictionRw.dtypes

trainRaw.isna().sum()

# Adding OutletSales column with nan values

predictionRw["OutletSales"]=np.nan
predictionRw.dtypes
predictionRw.shape

# sampling data into train & test

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(trainRaw,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# Add source column in both train ,test & prediction

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionRw["Source"]="Prediction"

trainDf.shape
testDf.shape
predictionRw.shape

# combine train,test & prediction datasets

fullRaw=pd.concat([trainDf,testDf,predictionRw],axis=0)
fullRaw.shape

# remove identifier column

fullRaw.drop(["ProductID","OutletID"],axis=1,inplace=True)
fullRaw.shape

# checking & treating NA values -Univariate Analysis

fullRaw.isna().sum()

# we have got 2 columns weight & OutletSize which has nan values

# 1st method

for i in fullRaw.columns:
    if i!="OutletSales":
        
        if (fullRaw.loc[:,i].dtype=="int64") |(fullRaw.loc[:,i].dtype=="float64"):
            if (fullRaw.loc[:,i].isna().sum())>0:
                tempMedian=fullRaw.loc[fullRaw["Source"]=="Train",i].median()
                fullRaw.loc[:,i].fillna(tempMedian,inplace=True)
                
        else:
            if (fullRaw.loc[:,i].isna().sum())>0:
                tempMode=fullRaw.loc[fullRaw["Source"]=="Train",i].mode()[0]
                fullRaw.loc[:,i].fillna(tempMode,inplace=True)
            
# 2nd method

for i in fullRaw.columns:
    if i!="OutletSales":
        
        if (fullRaw.loc[:,i].dtype=="int64") |(fullRaw.loc[:,i].dtype=="float64"):
            tempMedian=fullRaw.loc[fullRaw["Source"]=="Train",i].median()
            missingValueRows=fullRaw.loc[:,i].isna().sum()
            fullRaw.loc[:,i].fillna(tempMedian,inplace=True)
                
        else:
            tempMode=fullRaw.loc[fullRaw["Source"]=="Train",i].mode()[0]
            missingValueRows=fullRaw.loc[:,i].isna().sum()
            fullRaw.loc[:,i].fillna(tempMode,inplace=True)





# check the dtype of NA columns
            
fullRaw["Weight"].dtype
fullRaw["OutletSize"].dtype

#We have Weight is continuous variable

tempMedian=fullRaw.loc[fullRaw["Source"]=="Train","Weight"].median()
tempMedian 
fullRaw["Weight"].fillna(tempMedian,inplace=True)

# We have OutletSize which is categorical Variable and has nan value

tempMode=fullRaw.loc[fullRaw["Source"]=="Train","OutletSize"].mode()[0]
tempMode

fullRaw["OutletSize"].fillna(tempMode,inplace=True)

fullRaw.isna().sum()

# Bivariate Analysis -Correlation Matrix(continuous Variables)

corrDf=fullRaw[fullRaw["Source"]=="Train"].corr()
corrDf

sns.heatmap(corrDf,xticklabels=corrDf.columns,yticklabels=corrDf.columns,cmap="winter_r")

# correlation check for categorical variables-Boxplot(Dependent Vs Independent )

fullRaw.dtypes=="object"

sns.boxplot(y=trainDf["OutletSales"],x=trainDf["FatContent"])
sns.boxplot(y=trainDf["OutletSales"],x=trainDf["ProductType"])

sns.boxplot(y=trainDf["OutletSales"],x=trainDf["OutletSize"])

sns.boxplot(y=trainDf["OutletSales"],x=trainDf["LocationType"])

sns.boxplot(y=trainDf["OutletSales"],x=trainDf["OutletType"])

# Dummy Variable creation

fullRaw2=pd.get_dummies(fullRaw,drop_first=True)

#drop_first -it will ensure that we get n-1 dummies

fullRaw2.shape
fullRaw.shape

# Add intercept

from statsmodels.api import add_constant
fullRaw2=add_constant(fullRaw2)
fullRaw2.shape

# sampling data into train,test & prediction and drop source column

trainDf=fullRaw2[fullRaw2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1).copy()
testDf=fullRaw2[(fullRaw2["Source_Test"]==1)].drop(["Source_Train","Source_Test"],axis=1).copy()
predictionDf=fullRaw2[(fullRaw2["Source_Train"]==0) & (fullRaw2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Divide X(independents) & Y(dependents)
trainDf.columns
trainX=trainDf.drop(["OutletSales"],axis=1).copy()
trainY=trainDf["OutletSales"].copy()
testX=trainDf.drop(["OutletSales"],axis=1).copy()
testY=trainDf["OutletSales"].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape

# VIF check

from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF=5
MaxVIFcutoff=5
trainXcopy=trainX.copy()
counter=0
highMaxVIFcolumn=[]

while(tempMaxVIF>=MaxVIFcutoff):
    print(counter)
    
    #create an empty DataFrame to store VIF's
    tempVIFDf=pd.DataFrame()
    
    #Calculate VIF usingLIst comprehensions
    tempVIFDf["VIF"]=[variance_inflation_factor(trainXcopy.values,i) for i in range(trainXcopy.shape[1])]
    
    #create a column names against VIF values
    tempVIFDf["Column Name"]=trainXcopy.columns
    
    #drop na value
    tempVIFDf=tempVIFDf.dropna()
    #tempVIFDf.dropna(inplace=True)
    
    #sort the highest values in one variable based
    tempColumnName=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,1]
    
    # store the max values in VIF
    tempMaxVIF=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,0]
    
    print(tempColumnName)
    
    if(tempMaxVIF>=MaxVIFcutoff):
        trainXcopy=trainXcopy.drop(tempColumnName,axis=1)
        highMaxVIFcolumn.append(tempColumnName)

highMaxVIFcolumn

highMaxVIFcolumn.remove("const")
highMaxVIFcolumn

trainX=trainX.drop(highMaxVIFcolumn,axis=1)
testX=testX.drop(highMaxVIFcolumn,axis=1)
predictionDf=predictionDf.drop(highMaxVIFcolumn,axis=1)

trainX.shape
testX.shape
predictionDf.shape

# Model Building

from statsmodels.api import OLS
ModelDef=OLS(trainY,trainX)
Modelbuild=ModelDef.fit()

Modelbuild.summary()

dir(Modelbuild)
Modelbuild.rsquared
Modelbuild.rsquared_adj
Modelbuild.pvalues

#Significant variables selection

tempMaxpvalue=0.05
tempMaxpvalueCutoff=0.05
trainXcopy=trainX.copy()
counter=1
highMaxpvalueColumn=[]

while(tempMaxpvalue>=tempMaxpvalueCutoff):
    print(counter)
    Model=OLS(trainY,trainXcopy).fit()
    tempMaxDf=pd.DataFrame()
    tempMaxDf["Pvalues"]=Model.pvalues
    tempMaxDf["Column Name"]=trainXcopy.columns
     
    tempMaxDf.dropna(inplace=True)
    tempColumnName=tempMaxDf.sort_values(["Pvalues"],ascending=False).iloc[0,1]
    tempMaxpvalue=tempMaxDf.sort_values(["Pvalues"],ascending=False).iloc[0,0]
    
    
    if(tempMaxpvalue>=tempMaxpvalueCutoff):
        print(tempColumnName,tempMaxpvalue)
        trainXcopy=trainXcopy.drop(tempColumnName,axis=1)
        highMaxpvalueColumn.append(tempColumnName)
    
    counter=counter+1

highMaxpvalueColumn
    
# Drop insignificant variables

trainX=trainX.drop(highMaxpvalueColumn,axis=1)
testX=testX.drop(highMaxpvalueColumn,axis=1)
predictionDf=predictionDf.drop(highMaxpvalueColumn,axis=1)

trainX.shape
testX.shape
predictionDf.shape

#final model build after removing insignificant variables

Model=OLS(trainY,trainX).fit()
Model.summary()
Model.pvalues

#Model Prediction

Model_Pred=Model.predict(testX)

# Model Diagonistics

#homosekdasicity check- We check fitted values(predicted values) & 
# residuals variance check.It should show constant variance.
# it should not show any type of pattern like increasing or decreasing

import seaborn as sns

sns.scatterplot(Model.fittedvalues,Model.resid)


#Normality of residuals/errors check-

sns.distplot(Model.resid)

# RMSE-Root mean square error

RMSE_Value= np.sqrt(np.mean((testY-Model_Pred)**2))
RMSE_Value # 1588



# prediction

predictionDf["OutletSalesPrice"]=Model.predict(predictionDf.drop(["OutletSales"],axis=1))
predictionDf.to_csv("PredictionDf.csv")