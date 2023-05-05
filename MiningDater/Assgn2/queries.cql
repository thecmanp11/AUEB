//1
MATCH (art:article)-[r1:cites]-> (art2:article)
MATCH (aut:author)-[r2:writes]-> (art2:article)
RETURN aut.author, COUNT(r1) as citations
ORDER BY citations DESC
LIMIT 5;

//2
MATCH (a:author)-[:writes]->(b)
, ()-[r:writes]->(b)
return a, count(r) as collabs
ORDER BY collabs DESC
Limit 5;

//3
MATCH (a:author)-[:writes]-(b:article),()-[r:writes]-(b)
with a,b,count(r) as otherauthors
where otherauthors <= 1
RETURN a, count(b)
order by count(b) desc
Limit 1

//4
MATCH (a:author)-[r:writes]->(b:article)
where b.year = '2001'
RETURN a.author, count(r) as numbofpapers
order by numbofpapers desc
limit 1

//5
MATCH (b:article)
where b.year = '1998' and  b.title contains 'gravity'
return b.journal, count(*) as numbjourns
order by numbjourns desc

//6
MATCH ()-[r:cites]->(b:article)
return b, count(r) as citations
order by citations desc 
limit 5

//7
MATCH (b:article)<-[:writes]-(a:author)
where b.abstract contains 'holography' and b.abstract contains 'anti de sitter'
return a.author, b.title

//8
MATCH p = shortestPath((c:author{author:'Yuri Shirman'})-[*]-(f:author{author:'H. Lu'}))
RETURN [n in nodes(p) | n.author] AS ShortestPath, length(p) as Length

//9
MATCH p = shortestPath((c:author{author:'Yuri Shirman'})-[r:writes]-(f:author{author:'H. Lu'}))
RETURN [n in nodes(p) | n.author] AS ShortestPath, length(p) as Length

//10
MATCH p = (c:author{author:'Edward Witten'})-[:writes*25]-(f:author)
with f.author as author, length(p) as leng, f.title as title, relationships(p) as r
return author, leng, title
