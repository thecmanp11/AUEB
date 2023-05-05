import pandas as pd
import numpy as np  
import random

from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(sparse_output=True)


# PART 1::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Import initial dataset
df = pd.read_csv('groceries.csv',sep=';')
df['Customer_ID'] = df['Customer_ID'].astype('str')
# fill in blank Age values with average of column
df['Age'] = df['Age'].replace(' ',None).astype('float64')
df['Age'] = df['Age'].fillna(df['Age'].mean()).astype('int64')
# fill in blank Age values with average of column
df['Income'] = df['Income'].replace(' ',None).astype('float64')
df['Income'] = df['Income'].fillna(df['Income'].mean()).astype('int64')

# Create list of each customer's grocery items
groc = df['Groceries'].to_list()
groclist = []
for i in range(0,len(groc)):
    item = groc[i].split(sep=',')
    groclist.append(item)
# make a new column with each customers' groceries in a list
df['GroceriesList'] = groclist

# df is the final dataset



# PART 2::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# All Functions Take Customer_ID as input

# Categorical Attributes ---------------------------------------------------------
# simple evaluation of == or not between two customers
### Sex
sexl = df['Sex']
def SexComparison(a,b):
    a = a-1
    b = b-1
    if sexl[a] == sexl[b]:
        return 0
    else:
        return 1

### Marital Status
marrigl = df['Marital_Status']
def MarriageComparison(a,b):
    a = a-1
    b = b-1
    if marrigl[a] == marrigl[b]:
        return 0
    else:
        return 1

### Occupation
occupl = df['Occupation']
def OccupationComparison(a,b):
    a = a-1
    b = b-1
    if occupl[a] == occupl[b]:
        return 0
    else:
        return 1


# Ordinal Attributes ------------------------------------------------------------------
#### Must order, then convert categorical values to a numerical,
### Education
eduval = df['Education']
edudict = {'primary':1, 'secondary':2, 'tertiary':3}
edul = list((pd.Series(eduval)).map(edudict))
def EducationComparison(a,b):
    a = a-1
    b = b-1
    A = abs(edul[a]) - abs(edul[b])
    B = max(edul) - min(edul)
    C = A / B
    if C < 0:
        C = C * -1
    return C

### Customer_Rating
crval = df['Customer_Rating']
crdcit = {'poor':1, 'fair':2, 'good':3,'very_good':4,'excellent':5}
crl = list((pd.Series(crval)).map(crdcit))
def CustRatingComparison(a,b):
    A = abs(crl[a]) - abs(crl[b])
    B = max(crl) - min(crl)
    C = A / B
    if C < 0:
        C = C * -1
    return C

# Numerical Attributes -----------------------------------------------

### Age 
agl = df['Age']
def AgeComparison(a,b):
    a = a-1
    b = b-1
    A = abs(agl[a]) - abs(agl[b])
    B = max(agl) - min(agl)
    C = A / B
    if C < 0:
        C = C * -1
    return C

### Income
incl = df['Income']
def IncomeComparison(a,b):
    a = a-1
    b = b-1
    A = abs(incl[a]) - abs(incl[b])
    B = max(incl) - min(incl)
    C = A / B
    if C < 0:
        C = C * -1
    return C

### Persons_in_Household
ppl = df['Persons_in_Household']
def PplInHouseComparison(a,b):
    a = a-1
    b = b-1
    A = abs(ppl[a]) - abs(ppl[b])
    B = max(ppl) - min(ppl)
    C = A / B
    if C < 0:
        C = C * -1
    return C

# Set Attributes -----------------------------------------------------------------
### Groceries
# using Jaccard **DISS**imilarty!! aka Inverse of Jaccard Similarty
# so 100% similarty is converted to 0% dissimilarity, 75% similarity is converted to 25% dissimilarity, etc.
# |S1 intersetion S2|/|S1 union S2|
def GroceriesComparison(a, b):
    intersection = len(set(groclist[a]).intersection(set(groclist[b])))
    union = len(set(groclist[a]).union(set(groclist[b])))
    if int(intersection) / int(union) == 1:
        return 0
    else:
        return 1- (int(intersection) / int(union))


## Combining the scores --------------------------------------------------------------------------
#### How DISsimilar are these two customers
def ComboScores(a,b):
    sumry = SexComparison(a,b) + MarriageComparison(a,b) + OccupationComparison(a,b) + EducationComparison(a,b) + CustRatingComparison(a,b) + AgeComparison(a,b) + IncomeComparison(a,b) + PplInHouseComparison(a,b) + GroceriesComparison(a,b)
    return sumry / 9





# PART 3::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# create a function that takes a Customer_ID as input and returns the dissimilarty scores between them and all other customers
# then we just take the top 10 (i.e. the lowest dissimilarity values - a.k.a the most similar)
def Top10(a):
    scores = {}
    ind = a - 1
    for i in range(0,9999):
        cust = i + 1
        val = ComboScores(ind,i)
        scores.update( {'{} vs {}'.format(a, cust) : val} )
    data = pd.DataFrame(scores.items())
    data.columns = ['0','1']
    data[['Customer','Compared to']] = data['0'].str.split(" vs ",expand=True)
    data = data.sort_values(['1']).groupby('Customer').head(10)
    data.columns = ['Title','Dissimilarity Score','Customer','Compared To']
    print('10 NN for Customer ',a)
    return data

# For each customer in our list, run the function and append the data to the 'results' dataframe
custtoexamine = [73, 563, 1603, 2200, 3703, 4263, 5300, 6129, 7800, 8555]
results = pd.DataFrame()
for cust in custtoexamine:
    data = Top10(cust)
    results = results.append(data)

# 'results' is a dataframe with the top 10 NN for each customer in the given list



# PART 4::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Recreate ComboScores but without Customer Ranking
def NonCRScores(a,b):
    sumry = SexComparison(a,b) + MarriageComparison(a,b) + OccupationComparison(a,b) + EducationComparison(a,b) + AgeComparison(a,b) + IncomeComparison(a,b) + PplInHouseComparison(a,b) + GroceriesComparison(a,b)
    return sumry / 8

# this function gets us the top 10 most similar customers, then takes the average value of the Customer Rating for all those customers
def AvgRankPrediction(a):
    # get the top 10
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
    
    # Get all customer rating for the top 10
    crnks = []
    for i in range(0,len(data)):
        cust = data['Compared To'][i]
        ind = int(cust)
        rnk = df['Customer_Rating'][ind]
        crnks.append(rnk)
    crnks = list((pd.Series(crnks)).map(crdcit))
    
    # take the average of all those ratings
    avg = sum(crnks)/len(crnks)
    avg = int(round(avg,0))
    
    return avg

# this function gets us the top 10 most similar customers, 
# then takes the weighted average value of the Customer Rating for all those customers
def WAvgRankPrediction(a):
    # top 10
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
    
    # Get all customer rating for the top 10
    crnks = []
    for i in range(0,len(data)):
        cust = data['Compared To'][i]
        ind = int(cust)
        rnk = df['Customer_Rating'][ind]
        crnks.append(rnk)
    crnks = list((pd.Series(crnks)).map(crdcit))
    data['Cust Ranking'] = crnks
    
    # take the weighted average of those top 10
    ns = []
    ds = []
    for i in range(0,len(data)):
        n = data['Dissimilarity Score'][i] * (data['Cust Ranking'][i])
        ns.append(n)
        d = data['Dissimilarity Score'][i] 
        ds.append(d)
    wavg = sum(ns)/sum(ds)
    wavg = int(round(wavg,0))
    
    return wavg

# Prepare a dataset of the first 50 customers.
fiddy = df.head(50)
cust5 = fiddy['Customer_ID']

# for all those customers, run our Average Prediction function to get all of their predicted values
avg=[]
for cust in cust5:
    pred = AvgRankPrediction(int(cust))
    avg.append(pred)
# add a column for all predicted values
fiddy['Avg Prediction'] = avg

# for all those customers, run our Weighted Average Prediction function to get all of their predicted values
wavg= []
for cust in cust5:
    pred = WAvgRankPrediction(int(cust))
    wavg.append(pred)
# add a column for all predicted values
fiddy['W Avg Prediction'] = wavg

# Decode all those categorical ratings as a numerical value and create a column with this data
rting = fiddy['Customer_Rating']
crdcit = {'poor':1, 'fair':2, 'good':3,'very_good':4,'excellent':5}
decoded = list((pd.Series(rting)).map(crdcit))
fiddy['Num Rating'] = decoded
fiddy = fiddy.reset_index()

# Evaluate our Predictions

# Average Prediction MPE
ns = []
for i in range(0,len(fiddy)):
    n = int(fiddy['Avg Prediction'][i]) - int(fiddy['Num Rating'][i])
    ns.append(n)
AvgMPE = sum(ns) / len(fiddy)

# Weighted Average Prediction MPE
ns = []
for i in range(0,len(fiddy)):
    n = int(fiddy['W Avg Prediction'][i]) - int(fiddy['Num Rating'][i])
    ns.append(n)
WAvgMPE = sum(ns) / len(fiddy)

# display the results
print('Average MPE Score: ', AvgMPE, ', Weighted Average MPE Score: ' , WAvgMPE)
