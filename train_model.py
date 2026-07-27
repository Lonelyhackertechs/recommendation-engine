import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer



print("==============================")
print("Loading dataset")
print("==============================")



BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset.pkl"
)



MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)



os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



df = pd.read_pickle(
    DATASET_PATH
)



print(
    "Training records:",
    len(df)
)




if df.empty:

    raise Exception(
        "Dataset is empty. Add discussions first."
    )



df = df.fillna("")



df["content"] = (

    df["title"].astype(str)

    + " "

    + df["body"].astype(str)

    + " "

    + df["group_name"].astype(str)

    + " "

    + df["group_description"].astype(str)

)



print("Creating TF-IDF vectors...")



vectorizer = TfidfVectorizer(

    stop_words="english",

    ngram_range=(1,2),

    max_features=10000

)



vectors = vectorizer.fit_transform(

    df["content"]

)



print(
    "Saving trained model..."
)



with open(
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )



with open(
    os.path.join(
        MODEL_DIR,
        "discussion_vectors.pkl"
    ),
    "wb"
) as file:

    pickle.dump(
        vectors,
        file
    )



df[
    [
        "id",
        "title",
        "group_name"
    ]
].to_pickle(

    os.path.join(
        MODEL_DIR,
        "discussions.pkl"
    )

)



print("==============================")
print("MODEL TRAINED SUCCESSFULLY")
print("==============================")

print(
    "Discussions trained:",
    len(df)
)


print(
    "Models saved in:",
    MODEL_DIR
)
