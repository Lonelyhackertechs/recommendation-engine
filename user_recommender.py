import os
import pickle
import logging

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from database import get_connection


logging.basicConfig(level=logging.INFO)



# =====================================================
# LOAD DISCUSSION MODEL
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


with open(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    "rb"
) as file:

    vectorizer = pickle.load(file)



with open(
    os.path.join(MODEL_DIR, "discussion_vectors.pkl"),
    "rb"
) as file:

    discussion_vectors = pickle.load(file)



discussions = pd.read_pickle(
    os.path.join(MODEL_DIR, "discussions.pkl")
)



# =====================================================
# SETTINGS
# =====================================================

MIN_GROUP_MATCH = 0.05



# =====================================================
# USER DISCUSSION PROFILE
# =====================================================

def get_user_profile(user_id):

    engine = get_connection()


    try:

        query = """

        SELECT
            title,
            body

        FROM discussions

        WHERE user_id = %s


        UNION ALL


        SELECT
            d.title,
            d.body

        FROM replies r

        INNER JOIN discussions d

        ON d.id = r.discussion_id

        WHERE r.user_id = %s

        """


        data = pd.read_sql(

            query,

            engine,

            params=(

                user_id,

                user_id

            )

        )


        if data.empty:

            return None



        profile = []


        for _, row in data.iterrows():

            profile.append(

                f"{row['title']} {row['body']}"

            )


        return " ".join(profile)



    finally:

        engine.dispose()



# =====================================================
# DISCUSSION RECOMMENDATIONS
# =====================================================

def recommend_for_user(
        user_id,
        limit=5
):


    profile = get_user_profile(user_id)


    if not profile:

        return {

            "status": "cold_start",

            "recommendations": []

        }



    user_vector = vectorizer.transform(

        [profile]

    )



    scores = cosine_similarity(

        user_vector,

        discussion_vectors

    )[0]



    indexes = scores.argsort()[::-1]



    recommendations = []



    for index in indexes[:limit]:

        row = discussions.iloc[index]


        recommendations.append({

            "id":
            int(row["id"]),


            "title":
            row["title"],


            "group_name":
            row["group_name"],


            "similarity":
            round(

                float(scores[index]) * 100,

                2

            )

        })



    return {

        "status": "success",

        "recommendations": recommendations

    }





# =====================================================
# GROUP RECOMMENDATIONS
# =====================================================

def recommend_groups_for_user(
        user_id,
        limit=5
):


    engine = get_connection()


    try:


        with engine.connect() as conn:



            # -----------------------------------------
            # Build user interest profile from database
            # -----------------------------------------

            history = pd.read_sql(

                """

                SELECT

                    g.name AS title,

                    g.description AS body


                FROM groups g


                INNER JOIN group_members gm

                ON g.id = gm.group_id


                WHERE gm.user_id = %s



                UNION ALL



                SELECT

                    d.title,

                    d.body


                FROM discussions d


                WHERE d.user_id = %s



                UNION ALL



                SELECT

                    d.title,

                    r.body


                FROM replies r


                INNER JOIN discussions d

                ON d.id = r.discussion_id


                WHERE r.user_id = %s


                """,

                conn,

                params=(

                    user_id,

                    user_id,

                    user_id

                )

            )




            if history.empty:


                return {

                    "status":
                    "cold_start",

                    "recommendations":
                    []

                }




            # -----------------------------------------
            # Get groups not joined
            # -----------------------------------------

            groups = pd.read_sql(

                """

                SELECT

                    id,

                    name,

                    description


                FROM groups


                WHERE status='approved'


                AND id NOT IN

                (

                    SELECT group_id

                    FROM group_members

                    WHERE user_id=%s

                )


                """,

                conn,

                params=(user_id,)

            )




        if groups.empty:


            return {

                "status":
                "success",

                "recommendations":
                []

            }




        # -----------------------------------------
        # Create user text
        # -----------------------------------------

        user_text = " ".join(

            (

                history["title"].astype(str)

                +

                " "

                +

                history["body"].astype(str)

            ).tolist()

        )



        # -----------------------------------------
        # Create group text
        # -----------------------------------------

        group_text = (

            groups["name"].astype(str)

            +

            " "

            +

            groups["description"].astype(str)

        ).tolist()



        # -----------------------------------------
        # TF-IDF similarity
        # -----------------------------------------

        tfidf = TfidfVectorizer(

            stop_words="english",

            ngram_range=(1,2)

        )


        vectors = tfidf.fit_transform(

            [

                user_text

            ]

            +

            group_text

        )



        scores = cosine_similarity(

            vectors[0],

            vectors[1:]

        )[0]



        groups["score"] = scores




        # -----------------------------------------
        # Remove weak matches
        # -----------------------------------------

        groups = groups[

            groups["score"]

            >=

            MIN_GROUP_MATCH

        ]



        if groups.empty:


            return {

                "status":

                "success",

                "recommendations":

                []

            }




        # -----------------------------------------
        # Rank groups
        # -----------------------------------------

        groups = groups.sort_values(

            by="score",

            ascending=False

        ).head(limit)




        recommendations = []



        for _, group in groups.iterrows():


            recommendations.append({

                "id":
                int(group["id"]),


                "name":
                group["name"],


                "description":
                group["description"],


                "score":
                round(

                    float(group["score"]) * 100,

                    2

                ),


                "reason":

                "Recommended based on your activity"

            })




        logging.info(
            "User %s recommended groups: %s",
            user_id,
            recommendations
        )



        return {


            "status":

            "success",


            "recommendations":

            recommendations

        }



    finally:

        engine.dispose()