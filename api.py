from flask import Flask, jsonify
from user_recommender import recommend_for_user, recommend_groups_for_user
from calculate_score import run_batch_scoring
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

import atexit
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =====================================================
# Batch scoring scheduler
# =====================================================

def start_scheduler():
    """Start background scheduler for batch scoring.
    
    NOTE: For production, consider moving batch scoring to a separate
    worker process (Celery, etc.) to avoid blocking main app requests.
    """
    scheduler = BackgroundScheduler()
    
    # Run once shortly after startup
    scheduler.add_job(
        func=run_batch_scoring,
        trigger="date",
        run_date=datetime.now() + timedelta(seconds=5)
    )
    
    # Run every 5 minutes (reduced from 2 to decrease database load)
    scheduler.add_job(
        func=run_batch_scoring,
        trigger="interval",
        minutes=5
    )
    
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler

scheduler = start_scheduler()

# =====================================================
# Discussion recommendations
# =====================================================

@app.route("/recommendations/<int:user_id>")
def recommendations(user_id):
    """Get discussion recommendations for a user."""
    try:
        results = recommend_for_user(user_id)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error fetching recommendations for user {user_id}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": []
        }), 500

# =====================================================
# Group recommendations
# =====================================================

@app.route("/recommend-groups/<int:user_id>")
def recommend_groups(user_id):
    """Get group recommendations for a user."""
    try:
        results = recommend_groups_for_user(user_id, limit=5)
        return jsonify(results)
    
    except Exception as e:
        logger.exception(f"Error fetching group recommendations for user {user_id}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": []
        }), 500

@app.route("/")
def home():
    """Health check endpoint."""
    return jsonify({"status": "Recommendation API running"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
