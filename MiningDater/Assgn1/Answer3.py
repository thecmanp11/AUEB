import pandas as pd
import numpy as np  
import random

from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(sparse_output=True)

# create a function that takes a Customer_ID as input and returns the dissimilarty scores between them and all other customers
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

# For each customer in our list, run the function
custtoexamine = [73, 563, 1603, 2200, 3703, 4263, 5300, 6129, 7800, 8555]
results = pd.DataFrame()
for cust in custtoexamine:
    data = Top10(cust)
    results = results.append(data)

# 'results' is a dataframe with the top 10 NN for each customer in the given list