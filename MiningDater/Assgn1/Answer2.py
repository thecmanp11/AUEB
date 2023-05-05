import pandas as pd
import numpy as np  
import random

# for one hot encoding
from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(sparse_output=True)

# All Functions Take Customer_ID as input

# Categorical Attributes ---------------------------------------------------------

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
#### Must order, then convert categorical values to a numerical

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