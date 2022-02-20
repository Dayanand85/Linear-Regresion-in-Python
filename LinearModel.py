# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:32:21 2022

@author: Dayanand
"""

# Loading library

import os
import pandas as pd
import numpy as np
import seaborn as sns

# increase the display size
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# change directory
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\Job-a-thon")

# loading file train & test datasets.Calling test file to prediction dataset

rawDf=pd.read_csv("train_0OECtn8.csv")
predictionDf=pd.read_csv("test_1zqHu22.csv")

rawDf.shape
predictionDf.shape

# we have one more column in prediction then raw.Let us see columns
rawDf.columns
predictionDf.columns
# engagement_score column is more in prediction dataset.Let us add this column
predictionDf["engagement_score"]=0
predictionDf.shape

# Let us divide the rawDf into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)

trainDf.shape
testDf.shape

# Let us source column in train,test & prediction

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

# Let us combine all three datasets for data processing
fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullDf.shape

# let us drop identifier columns which are not of use

fullDf.columns
fullDf.drop(["row_id","user_id","category_id","video_id"],axis=1,inplace=True)
fullDf.shape

# let us check NULL values
fullDf.isna().sum() # No Null values

#Bivariate Analysis Continuous Variables:Scatter plot

corrDf=fullDf[fullDf["Source"]=="Train"].corr() #inference always shold be from Train data 
corrDf.head

sns.heatmap(corrDf,
            xticklabels=corrDf.columns,
            yticklabels=corrDf.columns,
            cmap='YlOrBr')

# Bivariate Analysis Categorical Variables:Boxplot
sns.boxplot(y=trainDf["engagement_score"],x=trainDf["gender"])
# Male is more engagement_score than female
sns.boxplot(y=trainDf["engagement_score"],x=trainDf["profession"])
# other and working_professional has almost same levele of engagement_score

# dummy variable creation
fullDf2=pd.get_dummies(fullDf,drop_first=False)
fullDf2.shape

# Add intercept column
from statsmodels.api import add_constant
fullDf2=add_constant(fullDf2)
fullDf2.shape

############################
# Divide the data into Train and Test
############################
# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column

# Step 1: Divide into Train and Testest

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy() 
predictDf=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()

########################
# Sampling into X and Y
########################

# Divide each dataset into Indep Vars and Dep var

depVar="engagement_score"
trainX=trainDf.drop([depVar],axis=1)
trainY=trainDf[depVar]

testX=testDf.drop([depVar],axis=1)
testY=testDf[depVar]

predictX=predictDf.drop([depVar],axis=1)

trainX.shape
trainY.shape
testX.shape
testY.shape
predictX.shape

# Model building
from statsmodels.api import OLS

M1_LR=OLS(trainY,trainX).fit()
M1_LR.summary()

# predict
Test1_Predict=M1_LR.predict(testX)

# Accuracy Check
from sklearn.metrics import r2_score
R2_Score1=r2_score(testY,Test1_Predict)
R2_Score1 # 0.26299

# Let us finalize significant variables

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

# Check final model summary

trainX = trainX.drop(highPValueColumnNames, axis = 1)
testX = testX.drop(highPValueColumnNames, axis = 1)
predictX = predictX.drop(highPValueColumnNames, axis = 1)

trainX.shape
testX.shape
predictX.shape
# Build model on trainX, trainY (after removing insignificant columns)
Model = OLS(trainY, trainX).fit()
Model.summary()

#########################
# Model Prediction
#########################

Test_Pred = Model.predict(testX)

# Let us checkr2_score
R2_Score2=r2_score(testY,Test_Pred)
R2_Score2 # 0.2629


# Prediction on PredictionDataSets

SampleSubmissionLR=pd.DataFrame()
SampleSubmissionLR["row_id"]=predictionDf["row_id"]
SampleSubmissionLR["engagement_score"]=Model.predict(predictX)
SampleSubmissionLR.to_csv("SampleSubmissionLR.csv",index=False)
