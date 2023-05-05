import pandas as pd
from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
conn = GraphDatabase.driver(uri, auth=("neo4j", "Ass2DB"), max_connection_lifetime=10000)

# Articles

art = pd.read_csv('Citation Dataset\\ArticleNodes.csv', header=None)#, nrows =10000)
art.columns = ['id','title','year','journal','abstract']
art['title'] = art['title'].astype(str)
art = art.replace(to_replace= r'\\', value= '', regex=True).replace(r'/','', regex=True).replace(r'"','', regex=True).replace(r"'",'', regex=True).replace(r'`','', regex=True).replace(r'{','', regex=True).replace(r'}','', regex=True).replace(r'\+','', regex=True)
art.head() # 29555

createlist = []
for index, row in art.iterrows():
    create = '(ID' + str(row['id']) + ":article { id: 'ID" + str(row['id']) + "', title: '" + str(row['title'])+ "', year: '" + str(row['year'])+ "', journal: '" + str(row['journal'])+ "'})"    # + "', abstract: '" + str(row['abstract']) 
    createlist.append(create)


createlist[0]

## Citation Relationships

cit = pd.read_csv('Citation Dataset\\Citations.csv', sep = '\t', header=None)#, nrows = 10000)
cit.columns = ['id1','id2']
cit['id1'] = cit['id1'].astype(str).str.strip()
cit['id2'] = cit['id2'].astype(str).str.strip()
cit.head()

for index, row in cit.iterrows():
    rel = '(ID' + row['id1'] + ')-[:cites]->(ID' + row['id2'] + ')'
    createlist.append(rel)

createlist[-1:]

# Author

aut = pd.read_csv('Citation Dataset\\AuthorNodes.csv', header=None)#, nrows=10000)
aut = aut.reset_index()
aut.columns = ['autid','id','author']
aut['autid'] = aut['autid'].astype(str).str.strip()
aut['id'] = aut['id'].astype(str).str.strip()
aut = aut.replace(to_replace= r'\\', value= '', regex=True).replace(r'/','', regex=True).replace(r'"','', regex=True).replace(r"'",'', regex=True).replace(r'`','', regex=True).replace(r'{','', regex=True).replace(r'}','', regex=True).replace(r'\+','', regex=True)
aut.head() 

for index, row in aut.iterrows():
    create = '(aID' + str(row['autid']) + ":author { id: 'ID" + str(row['id']) + "', author: '" + str(row['author']) + "'})" 
    createlist.append(create)
    
createlist[-1:]

for index, row in aut.iterrows():
    rel = '(aID' + row['autid'] + ')-[:writes]->(ID' + row['id'] + ')'
    createlist.append(rel)
    
createlist[-1:]


# List to string with commas
createstring = ','.join(createlist)
createstring = 'CREATE ' + createstring
createstring


# CREATE!!

with conn.session() as graphDB_Session:
    # Create nodes
    graphDB_Session.run(createstring)