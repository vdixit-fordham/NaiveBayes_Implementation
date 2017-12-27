import numpy as np
import pandas as pd
from pandas import DataFrame as df


def getProbForY(testRowDF , yValue , trainDF, trainLabelDF):
    #print(trainDF)
    featureList = list(trainDF.columns)
    
    columnCount = 0
    probability = 1
    for row in testRowDF.itertuples(index=False):
        print('*********** ', row._0)
        #print('########### ', len(trainDF[trainDF[featureList[columnCount]] == row._0]))
        #print(featureList[len(featureList) - 1])
        #print('$$$$$$$$$$ ', len(trainDF[trainDF[featureList[len(featureList) - 1]] == yValue]))
        totalYCount = len(trainDF[trainDF[featureList[len(featureList) - 1]] == yValue])
        totalMatchingXCount = len(trainDF[trainDF[featureList[len(featureList) - 1]] == yValue][trainDF[featureList[columnCount]] == row._0])
        
        #print('totalMatchingXCount' , totalMatchingXCount)
        #print('totalYCount' , totalYCount)
        
        distinctValues = len(trainDF[featureList[columnCount]].drop_duplicates())
        
        #print('&&&&&&&&&&&&&&&&&&&&&&&77 ', trainDF[featureList[columnCount]])
        #print('++++++++++++++++++++++++++++++++++ ', len(trainDF[featureList[columnCount]].drop_duplicates()))
        
        totalMatchingXCountAfterSmooth = totalMatchingXCount + 1
        totalYCountAfterSmooth = totalYCount + distinctValues
        #print('totalMatchingXCountAfterSmooth' , totalMatchingXCountAfterSmooth)
        #print('totalYCountAfterSmooth' , totalYCountAfterSmooth)
        #print(trainDF[trainDF[featureList[len(featureList) - 1]] == yValue][trainDF[featureList[columnCount]] == row._0])
        
        tmp = totalMatchingXCountAfterSmooth / totalYCountAfterSmooth
        
        probability = probability * tmp
        
        columnCount += 1
        
    priorProbForY = len(trainDF[trainDF[featureList[len(featureList) - 1]] == yValue]) / len(trainDF[featureList[len(featureList) - 1]])
    #print('^^^^^^^^^^^^^^^^^^ ', priorProbForY)
    
    
    return probability*priorProbForY  
        
        
        
trainDF = pd.read_csv("nv-train.csv")
testDF = pd.read_csv("nv-valiadation.csv")
#print(type(trainDF))
#print(type(testDF))

trainDF.drop('Instance', axis=1, inplace=True)
trainLabelDF = trainDF['Salary'].copy
testDF.drop('Instance', axis=1, inplace=True)

#print(trainDF)
#print(trainLabelDF)
#print(len(trainDF[trainDF['Salary'] == 'High']))
#print(len(trainDF['Salary']))
#print(trainLabelDF)

counter = 0

for row in testDF.itertuples(index=False):
        #print(row.Career)
        #print(type(row))
        #print(row.getcolumns)
        rowDF = pd.Series(row).to_frame()
        #print(rowDF)
        counter += 1
        
        print('******************************************************************************************')
        probForYHigh = getProbForY(rowDF, 'High', trainDF, trainLabelDF)
        print('\n\n\n')
        probForYLow = getProbForY(rowDF, 'Low', trainDF, trainLabelDF)
        print('******************************************************************************************')
        
        print('Probability for High Salary @@@@@@@@@@@@@@@ ' , probForYHigh*100)
        print('Probability for Low Salary  ############### ' , probForYLow*100)
        
        if(probForYHigh > probForYLow):
            print("Predicted Class Lebel is High")
        else:
            print("Predicted Class Lebel is Low")
