import os
import pickle
import logging
from functools import lru_cache
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from database import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# LOAD DISCUSSION MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as file:
    vectorizer = pickle.load(file)

with open(os.path.join(MODEL_DIR, "discussion_vectors.pkl"), "rb") as file:
    discussion_vectors = pickle.load(file)

discussions = pd.read_pickle(os.path.join(MODEL_DIR, "discussions.pkl"))

# Pre-compute discussion vector indices for faster lookup
discussion_indices = {int(row["id"]): idx for idx, (_, row) in enumerate(discussions.iterrows())}

# =====================================================
# SETTINGS
# =====================================================

MIN_GROUP_MATCH = 0.05
PROFILE_CACHE_TTL = 3600  # Cache user profiles for 1 hour
_profile_cache = {}  # {user_id: (profile_text, timestamp)}

# =====================================================
# USER DISCUSSION PROFILE
# =====================================================

def _get_cached_profile(user_id):
    """Check if a cached profile is still valid."""
    if user_id in _profile_cache:
        profile_text, timestamp = _profile_cache[user_id]
        if datetime.now() - timestamp < timedelta(seconds=PROFILE_CACHE_TTL):
            return profile_text
        else:
            del _profile_cache[user_id]
    return None

def get_user_profile(user_id):
    """Fetch and cache user discussion profile."""
    # Check cache first
    cached = _get_cached_profile(user_id)
    if cached is not None:
        return cached
    
    engine = get_connection()
    
    try:
        query = """
        SELECT title, body
        FROM discussions
        WHERE user_id = %s
        
        UNION ALL
        
        SELECT d.title, d.body
        FROM replies r
        INNER JOIN discussions d ON d.id = r.discussion_id
        WHERE r.user_id = %s
        """
        
        data = pd.read_sql(query, engine, params=(user_id, user_id))
        
        if data.empty:
            return None
        
        # Vectorized string concatenation (much faster than iterrows)
        profile = " ".join(
            (data['title'].astype(str) + " " + data['body'].astype(str)).tolist()
        )
        
        # Cache the profile
        _profile_cache[user_id] = (profile, datetime.now())
        
        return profile
    
    finally:
        engine.dispose()

# =====================================================
# DISCUSSION RECOMMENDATIONS
# =====================================================

def recommend_for_user(user_id, limit=5):
    """Recommend discussions for a user based on their profile."""
    profile = get_user_profile(user_id)
    
    if not profile:
        return {"status": "cold_start", "recommendations": []}
    
    # Vectorize user profile
    user_vector = vectorizer.transform([profile])
    
    # Compute cosine similarity against all discussions
    scores = cosine_similarity(user_vector, discussion_vectors)[0]
    
    # Get top-k indices using partial sort (more efficient than full sort)
    top_indices = np.argsort(-scores)[:limit]
    
    recommendations = []
    for idx in top_indices:
        row = discussions.iloc[idx]
        recommendations.append({
            "id": int(row["id"]),
            "title": row["title"],
            "group_name": row["group_name"],
            "similarity": round(float(scores[idx]) * 100, 2)
        })
    
    return {"status": "success", "recommendations": recommendations}

# =====================================================
# GROUP RECOMMENDATIONS
# =====================================================

def recommend_groups_for_user(user_id, limit=5):
    """Recommend groups for a user based on their activity and interests."""
    engine = get_connection()
    
    try:
        with engine.connect() as conn:
            # Build user interest profile from database
            history_query = """
            SELECT g.name AS title, g.description AS body
            FROM groups g
            INNER JOIN group_members gm ON g.id = gm.group_id
            WHERE gm.user_id = %s
            
            UNION ALL
            
            SELECT d.title, d.body
            FROM discussions d
            WHERE d.user_id = %s
            
            UNION ALL
            
            SELECT d.title, r.body
            FROM replies r
            INNER JOIN discussions d ON d.id = r.discussion_id
            WHERE r.user_id = %s
            """
            
            history = pd.read_sql(history_query, conn, params=(user_id, user_id, user_id))
            
            if history.empty:
                return {"status": "cold_start", "recommendations": []}
            
            # Get groups not joined (with LIMIT at SQL layer)
            groups_query = """
            SELECT id, name, description
            FROM groups
            WHERE status = 'approved'
            AND id NOT IN (SELECT group_id FROM group_members WHERE user_id = %s)
            LIMIT 100
            """
            
            groups = pd.read_sql(groups_query, conn, params=(user_id,))
            
            if groups.empty:
                return {"status": "success", "recommendations": []}
            
            # Vectorized text creation
            user_text = " ".join(
                (history['title'].astype(str) + " " + history['body'].astype(str)).tolist()
            )
            
            group_texts = (
                groups['name'].astype(str) + " " + groups['description'].astype(str)
            ).tolist()
            
            # TF-IDF similarity
            tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            vectors = tfidf.fit_transform([user_text] + group_texts)
            
            scores = cosine_similarity(vectors[0], vectors[1:])[0]
            groups["score"] = scores
            
            # Filter weak matches
            groups = groups[groups["score"] >= MIN_GROUP_MATCH]
            
            if groups.empty:
                return {"status": "success", "recommendations": []}
            
            # Rank groups and limit results
            groups = groups.sort_values(by="score", ascending=False).head(limit)
            
            recommendations = []
            for _, group in groups.iterrows():
                recommendations.append({
                    "id": int(group["id"]),
                    "name": group["name"],
                    "description": group["description"],
                    "score": round(float(group["score"]) * 100, 2),
                    "reason": "Recommended based on your activity"
                })
            
            logger.info("User %s recommended groups: %s", user_id, recommendations)
            
            return {"status": "success", "recommendations": recommendations}
    
    finally:
        engine.dispose()
