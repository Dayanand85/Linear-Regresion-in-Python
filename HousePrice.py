# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:06:11 2021

@author: Dayanand
"""
# loading library
 
import pandas as pd
import numpy as np
import seaborn as sns
import os

# Display output size

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# loading database

rawDf=pd.read_csv("C://Users//tk//Desktop//DataScience//Python Class Notes//PropertyPrice_Data.csv")
predictionDf=pd.read_csv("C://Users//tk//Desktop//DataScience//Python Class Notes//PropertyPrice_Prediction.csv")

# sampling the data into train,test &  prediction

from sklearn.model_selection import train_test_split

trainDf,testDf=train_test_split(rawDf,train_size=0.8,random_state=2410) 

trainDf.shape
testDf.shape

#Create source column in train,test & prediction

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

#Combine train & test data

fullRaw=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullRaw.shape

#Let us drop ID column

fullRaw=fullRaw.drop(["Id"],axis=1)
fullRaw.shape
fullRaw.columns

# Check NA value

fullRaw.isna().sum()

# Object column Garage & numeric column Garage_Built_Year has NA values

# check dtypes

fullRaw.dtypes

#Univariate Analysis with missing value & computation

tempMode=fullRaw.loc[fullRaw["Source"]=="Train","Garage"].mode()[0]#inference based on train
tempMode
fullRaw["Garage"].fillna(tempMode,inplace=True)#Action on whole datasets

tempMedian=fullRaw.loc[fullRaw["Source"]=="Train","Garage_Built_Year"].median()
tempMedian

fullRaw["Garage_Built_Year"].fillna(tempMedian,inplace=True)

fullRaw.isna().sum()

# Bivariate Analysis with continuous vars-Correlation -Scatterplot
# we do above analysis with independet to dependent as well as dependent to dependent vars
corrDf=fullRaw[fullRaw["Source"]=="Train"].corr()
corrDf

sns.heatmap(corrDf,xticklabels=corrDf.columns,yticklabels=corrDf.columns,cmap="winter_r")

#Bivariate Analysis with categorrical Variables-Boxplot
# We do above analysis only with depenent vs independent

categor_vars=trainDf.columns[trainDf.dtypes=="object"]
categor_vars

#Option One-Compare one by one

sns.boxplot(y=fullRaw.loc[fullRaw["Source"]=="Train","Sale_Price"],x=fullRaw.loc[fullRaw["Source"]=="Train","Road_Type"])
sns.boxplot(y=trainDf["Sale_Price"],x=trainDf["Property_Shape"])

# Option 2-do for loop

from matplotlib.pyplot import figure
for colName in categor_vars:
    figure() 
    sns.boxplot(y=trainDf["Sale_Price"],x=trainDf[colName])
 
#Option 3- run for loop and dump the plots in pdf
from matplotlib.backends.backend_pdf import PdfPages
fileName="C:/Users/tk/Desktop/DataScience/Python Class Npeotes/Categorical_Var_Analysis.pdf"
pdf=PdfPages(fileName)
for colNum,colName in enumerate(categor_vars):
    figure()
    print(colNum,colName)
    sns.boxplot(y=trainDf["Sale_Price"],x=trainDf[colName])
    pdf.savefig(colNum+1)
pdf.close()

#######################
#### Dummy_Variable_Creation
#######################

fullRaw2=pd.get_dummies(fullRaw,drop_first=True)

fullRaw2.shape
fullRaw.shape

################
### Add intercept names
################

from statsmodels.api import add_constant
fullRaw2=add_constant(fullRaw2)
fullRaw2.shape

##################
### Sampling
##################
# Step1: divide fulldata in trainDf,testDf & predictionDf

trainDf=fullRaw2[fullRaw2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1).copy()
testDf=fullRaw2[fullRaw2["Source_Test"]==1].drop(["Source_Train","Source_Test"],axis=1).copy()
predictionDf=fullRaw2[(fullRaw2["Source_Train"]==0) & (fullRaw2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Step 2: Divide Independent(X) and Dependent(Y) variables
trainX=trainDf.drop(["Sale_Price"],axis=1).copy()
trainY=trainDf["Sale_Price"].copy()
testX=testDf.drop(["Sale_Price"],axis=1).copy()
testY=testDf["Sale_Price"].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape
predictionDf.shape

#########################
##### VIF check
#########################

from statsmodels.stats.outliers_influence import variance_inflation_factor
tempMaxVIF=5
maxVIFcutoff=5
trainXCopy=trainX.copy()
counter=1
highVIFColumnNames=[]

while(tempMaxVIF>=maxVIFcutoff):
    print(counter)
    tempVIFDf=pd.DataFrame()
    tempVIFDf["VIF"]=[variance_inflation_factor(trainXCopy.values,i) for i in range(trainXCopy.shape[1])]
    tempVIFDf["Columns"]=trainXCopy.columns
    tempVIFDf.dropna(inplace=True)
    tempColumnName=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,1]
    tempMaxVIF=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,0]
    print(tempColumnName)
    
    if(tempMaxVIF>=maxVIFcutoff):
        trainXCopy=trainXCopy.drop(tempColumnName,axis=1)
        highVIFColumnNames.append(tempColumnName)
        counter=counter+1

highVIFColumnNames

highVIFColumnNames.remove("const")
highVIFColumnNames


trainX=trainX.drop(highVIFColumnNames,axis=1)
testX=testX.drop(highVIFColumnNames,axis=1)
predictionDf=predictionDf.drop(highVIFColumnNames,axis=1)

trainX.shape
testX.shape
predictionDf.shape

# Model Building
from statsmodels.api import OLS
Modeldef=OLS(trainY,trainX)
mBuildModel=Modeldef.fit()
mBuildModel.summary()

# selecting significant variables using while loop

tempMaxPValue = 0.1
maxPValueCutoff = 0.1
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    print(counter)    
    tempModelDf = pd.DataFrame()    
    Model = OLS(trainY, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) # If there is some calculation error resulting in NAs
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames

#check final model summary
Model.summary()
trainX=trainX.drop(highPValueColumnNames,axis=1)
testX=testX.drop(highPValueColumnNames,axis=1)
predictionDf=predictionDf.drop(highPValueColumnNames,axis=1)

# Building Model
Model=OLS(trainY,trainX).fit()
Model.summary()

# Prediction
Test_Pred=Model.predict(testX)

# Model Diagnostic Plots
dir(Model)
import seaborn as sns

sns.scatterplot(Model.fittedvalues,Model.resid)

# Normality of error check
sns.distplot(Model.resid)

#RMSE-Root mean square root error

np.sqrt(np.mean((testY-Test_Pred)**2))#45272
Model


# This means on an "average", the house price prediction would have +/- 
# error of about 45272

#MAPE-Mean absolute percentage error

(np.mean(np.abs(testY-Test_Pred)/testY))*100 # 19.85

# Generally, MAPE under 10% is considered very good, and anything under 20% is 
# reasonable.MAPE over 20% is usually not considered great.

# Now, is this a good model? Probably an "Average" model. If I told you your house was going to sell for $300,000 and 
# then it actually only sold for $244,000 (Roughly $56,000 error), you would be pretty mad. $56,000 is a 
# reasonable difference when you're buying/selling a home. 
# if the prediction is $300000, then the house would be cold somewhere between $244000 and $356000 
# But what if I told you that I could predict GDP (Gross Domestic Product) 
# of the US with only an average error of $56000? Well, since the GDPs are usually around $20 trillion, 
# that difference (of $56,000) wouldn't be so big. So, an RMSE of $56,000 would be acceptable in a GDP model but not 
# in Property prediction model like ours! So, there is a bit of relativity involved in RMSE values.

# Model Prediction (csv file)
Model_Prediction["Predicted_Sale_Prcie"] = Model.predict(predictionDf.drop(["Sale_Price"],axis=1))
predictionDf.to_csv("Model_Prediction.csv")
