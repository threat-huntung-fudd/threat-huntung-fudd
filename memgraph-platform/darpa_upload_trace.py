from neo4j import GraphDatabase
import csv
import pandas as pd

# Define correct URI and AUTH arguments (no AUTH by default)
URI = "bolt://127.0.0.1:7687"
AUTH = ("user", "changeme")

columns = [
    "exec",
    "location",
    "sequence",
    "edge_label",
    "source_type",
    "destination_type",
    "event_type",
    "ts",
]
df = pd.read_csv(
    "./create_graph/output_data/trace/darpa_trace/events_tgb_normal.csv",
    header=None,
    names=columns,
)


# create a nodelist start!!
exec_nodes = df[["exec", "source_type"]]
# Rename the 'source_type' column to 'node_type'
exec_nodes.rename(columns={"source_type": "node_type"}, inplace=True)
exec_nodes.rename(columns={"exec": "identificator"}, inplace=True)
location_nodes = df[["location", "destination_type"]]
location_nodes.rename(columns={"destination_type": "node_type"}, inplace=True)
location_nodes.rename(columns={"location": "identificator"}, inplace=True)

# Merge exec_nodes and location_nodes
merged_nodes = pd.concat([exec_nodes, location_nodes])

# Remove duplicate rows from the dataset

merged_nodes.drop_duplicates(inplace=True)

# Save the nodelist to a CSV file
merged_nodes.to_csv(
    "./create_graph/output_data/trace/events_tgb_normal_nodelist_v2.csv",
    index=False,
    header=False,
)


df.to_csv(
    "./create_graph/output_data/trace/events.csv",
    index=True,
    header=False,
)


with GraphDatabase.driver(URI, auth=AUTH) as client:
    # Check the connection
    client.verify_connectivity()

    records, summary, keys = client.execute_query(
        'LOAD CSV FROM "fudd/create_graph/output_data/trace/events_tgb_normal_nodelist_v2.csv" NO HEADER AS row CREATE (n:Node {id: row[0], type: row[1], value:row[0]});',
        database_="memgraph",
    )
    print("Loaded nodes")

    with client.session(
        database="memgraph"
    ) as session:  
        session.run("CREATE INDEX ON :Node(id)")
    print("Created index on nodes")

    records, summary, keys = client.execute_query(
        """LOAD CSV FROM "fudd/create_graph/output_data/trace/events.csv" NO HEADER AS row
            MATCH (n1:Node {id: row[1]}), (n2:Node {id: row[2]})
            CREATE (n1)-[event:rel]->(n2)
            SET	event.ts= toInteger(row[8]),
                event.type = row[7],
                event.label = row[4],
                event.indexid = row[0];
        """,
        database_="memgraph",
    )
    print("Loaded edges")

