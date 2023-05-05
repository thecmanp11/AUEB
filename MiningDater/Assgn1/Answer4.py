import pandas as pd
import numpy as np  
import random

# for one hot encoding
from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(sparse_output=True)

# Recreate ComboScores but without Customer Ranking
def NonCRScores(a,b):
    sumry = SexComparison(a,b) + MarriageComparison(a,b) + OccupationComparison(a,b) + EducationComparison(a,b) + AgeComparison(a,b) + IncomeComparison(a,b) + PplInHouseComparison(a,b) + GroceriesComparison(a,b)
    return sumry / 8


# Define a function which takes Customer_ID as input and OUtputs predicted Customer_Rating
def RankPrediction(a):
    scores = {}
    ind = a - 1
    for i in range(0,9999):
        cust = i + 1
        val = NonCRScores(ind,i)
        scores.update( {'{} vs {}'.format(a, cust) : val} )
    data = pd.DataFrame(scores.items())
    data.columns = ['0','1']
    data[['Customer','Compared to']] = data['0'].str.split(" vs ",expand=True)
    data = data.sort_values(['1']).groupby('Customer').head(10)
    data.columns = ['Title','Dissimilarity Score','Customer','Compared To']
    data = data.reset_index()
    
    crnks = []
    for i in range(0,len(data)):
        cust = data['Compared To'][i]
        ind = int(cust)
        rnk = df['Customer_Rating'][ind]
        crnks.append(rnk)
    crnks = list((pd.Series(crnks)).map(crdcit))
    data['Cust Ranking'] = crnks
    
    ns = []
    ds = []
    for i in range(0,len(test64)):
        n = data['Dissimilarity Score'][i] * (data['Cust Ranking'][i])
        ns.append(n)
        d = data['Dissimilarity Score'][i] 
        ds.append(d)
    wavg = sum(ns)/sum(ds)
    wavg = int(round(wavg,0))
    
    return wavg

# Create a dataframe with a random sample of 50 customers from our original dataset
rand5 = df.sample(50)
cust5 = rand5['Customer_ID']
predz= []
for cust in cust5:
    pred = RankPrediction(int(cust))
    predz.append(pred)
# add a column for all predicted values
rand5['Prediction'] = predz
# Convert Ordinal Rating into Numerical
rting = rand5['Customer_Rating']
crdcit = {'poor':1, 'fair':2, 'good':3,'very_good':4,'excellent':5}
decoded = list((pd.Series(rting)).map(crdcit))
rand5['Num Rating'] = decoded
rand5 = rand5.reset_index()
# For each customer in our dataframe of 50 random customers, evaluate and return MPE Score
ns = []
for i in range(0,len(rand5)):
    n = int(rand5['Prediction'][i]) - int(rand5['Num Rating'][i])
    ns.append(n)
mpe = sum(ns) / len(rand5)

mpe









