import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import Text2CypherRetriever

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create LLM 
t2c_llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={"temperature":0}
)

# Create Examples to feed retriever
examples = [
    """
    MATCH (node)<-[r:RATED]-()
    RETURN
        node.title AS title, node.plot AS plot, score AS similarityScore,
        collect { match (node)-[:IN_GENRE]->(g) RETURN g.name } as genres,
        collect { match (node)<-[:ACTED_IN]->(a) RETURN a.name} as actors,
        avg(r.rating) as userRating
    ORDER BY userRating DESC
    """
]

# Create Schema to narrow down scope of Graph
# This helps maintain the relationship between nodes on complex graphs.
neo4j_sheme = """
Node properties:
Person {name: STRING, born: INTEGER}
Movie {tagline: STRING, title: STRING, released: INTEGER}
Genre {name: STRING}
User {name: STRING}

Relationship properties:
ACTED_IN {role: STRING}
RATED {rating: INTEGER}

The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:DIRECTED]->(:Movie)
(:User)-[:RATED]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
"""


# Build the retriever
retriever = Text2CypherRetriever(
    driver=driver,
    llm=t2c_llm,
    examples = examples,
    neo4j_schema=neo4j_sheme,
)


llm = OpenAILLM(model_name="gpt-4o")
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = input("Input your movie Query:")

response = rag.search(
    query_text=query_text,
    return_context=True
    )

print(response.answer)
print("CYPHER :", response.retriever_result.metadata["cypher"])
print("CONTEXT:", response.retriever_result.items)

driver.close()
