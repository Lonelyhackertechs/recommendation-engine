from flask import Flask, jsonify
from user_recommender import (
    recommend_for_user,
    recommend_groups_for_user
)

from calculate_score import run_batch_scoring

from apscheduler.schedulers.background import BackgroundScheduler

import atexit
import os
import sys
import logging


sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


logging.basicConfig(
    level=logging.INFO
)


app = Flask(__name__)



# -----------------------------
# Batch scoring scheduler
# -----------------------------

def start_scheduler():

    run_batch_scoring()


    scheduler = BackgroundScheduler()


    scheduler.add_job(
        func=run_batch_scoring,
        trigger="interval",
        minutes=2
    )


    scheduler.start()


    atexit.register(
        lambda: scheduler.shutdown()
    )


    return scheduler



scheduler = start_scheduler()



# -----------------------------
# Discussion recommendations
# -----------------------------

@app.route("/recommendations/<int:user_id>")
def recommendations(user_id):

    try:

        results = recommend_for_user(
            user_id
        )

        return jsonify(results)


    except Exception as e:

        logging.exception(e)

        return jsonify({

            "status": "error",

            "message": str(e),

            "recommendations": []

        }), 500





# -----------------------------
# Group recommendations
# -----------------------------

@app.route("/recommend-groups/<int:user_id>")
def recommend_groups(user_id):

    try:

        results = recommend_groups_for_user(
            user_id,
            limit=5
        )


        return jsonify(results)


    except Exception as e:

        logging.exception(e)


        return jsonify({

            "status": "error",

            "message": str(e),

            "recommendations": []

        }), 500

@app.route("/")
def home():

    return jsonify({

        "status":
        "Recommendation API running"

    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )