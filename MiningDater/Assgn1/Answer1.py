import pandas as pd
import numpy as np  
import random

from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(sparse_output=True)

# Import initial dataset

df = pd.read_csv('groceries.csv',sep=';')
df['Customer_ID'] = df['Customer_ID'].astype('str')
df['Age'] = df['Age'].replace(' ',None).astype('float64')
df['Age'] = df['Age'].fillna(df['Age'].mean()).astype('int64')

df['Income'] = df['Income'].replace(' ',None).astype('float64')
df['Income'] = df['Income'].fillna(df['Income'].mean()).astype('int64')

# Create list of each customer's grocery items

groc = df['Groceries'].to_list()
groclist = []
for i in range(0,len(groc)):
    item = groc[i].split(sep=',')
    groclist.append(item)

df['GroceriesList'] = groclist

# df is the final dataset