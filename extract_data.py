import pandas as pd
import os
from database import get_connection



engine = get_connection()


query = """

SELECT

d.id,
d.title,
d.body,

g.name AS group_name,

g.description AS group_description


FROM discussions d


INNER JOIN groups g

ON d.group_id = g.id


"""


print("Loading database data...")


df = pd.read_sql(
    query,
    engine
)


print("Rows found:")
print(len(df))


print(df.head())



if df.empty:

    print(
        "NO DISCUSSIONS FOUND"
    )

    exit()



df = df.fillna("")



df["content"] = (

    df["title"]
    + " "
    + df["body"]
    + " "
    + df["group_name"]
    + " "
    + df["group_description"]

)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset.pkl"
)


df.to_pickle(
    DATASET_PATH
)



print(
"Dataset created successfully"
)

print(
"Saved records:",
len(df)
)
