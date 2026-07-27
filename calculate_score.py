import mysql.connector
from datetime import datetime
import math
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

WEIGHTS = {'discussion': 1, 'reply': 1, 'group_join': 2, 'group_message': 2}
DECAY_RATE = 0.02

def to_datetime(val):
    """Convert database datetime to Python datetime."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
    return datetime(2000, 1, 1)

def run_batch_scoring():
    """Run batch scoring in a separate worker process (not on main thread)."""
    missing = [k for k, v in DB_CONFIG.items() if v in (None, '')]
    if missing:
        logger.error(
            f"Batch scoring skipped — missing DB config values: {missing}. "
            f"Check that these are set as environment variables."
        )
        return
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.errors.DatabaseError as e:
        logger.error(f"Batch scoring skipped — could not connect to DB: {e}")
        return
    
    try:
        cursor = conn.cursor(dictionary=True)
        logger.info("Starting batch score calculation...")
        
        # Query uses indexes on user_id for efficient joining
        cursor.execute("""
            SELECT u.id, 
                   COUNT(DISTINCT d.id) as discussions,
                   COUNT(DISTINCT r.id) as replies,
                   COUNT(DISTINCT gm.group_id) as groups_joined,
                   COUNT(DISTINCT gms.id) as group_messages,
                   GREATEST(IFNULL(MAX(d.created_at), '2000-01-01 00:00:00'), 
                            IFNULL(MAX(r.created_at), '2000-01-01 00:00:00'), 
                            IFNULL(MAX(gm.created_at), '2000-01-01 00:00:00'),
                            IFNULL(MAX(gms.created_at), '2000-01-01 00:00:00'),
                            IFNULL(MAX(u.updated_at), '2000-01-01 00:00:00')) as last_active
            FROM users u
            LEFT JOIN discussions d ON d.user_id = u.id
            LEFT JOIN replies r ON r.user_id = u.id  
            LEFT JOIN group_members gm ON gm.user_id = u.id
            LEFT JOIN group_messages gms ON gms.user_id = u.id
            GROUP BY u.id
        """)
        
        users = cursor.fetchall()
        logger.info(f"Processing {len(users)} users for score calculation")
        
        for user in users:
            base = (
                user['discussions'] * WEIGHTS['discussion'] + 
                user['replies'] * WEIGHTS['reply'] +
                user['groups_joined'] * WEIGHTS['group_join'] +
                user['group_messages'] * WEIGHTS['group_message']
            )
            
            last_active = to_datetime(user['last_active'])
            days_inactive = (datetime.now() - last_active).days
            score = round(base * math.exp(-DECAY_RATE * max(0, days_inactive)), 2)
            
            cursor.execute(
                "INSERT INTO user_scores (user_id, score, updated_at) VALUES (%s, %s, NOW()) "
                "ON DUPLICATE KEY UPDATE score = %s, updated_at = NOW()",
                (user['id'], score, score)
            )
        
        conn.commit()
        logger.info("Batch score calculation completed successfully")
    
    except Exception as e:
        logger.exception(f"Batch scoring failed: {e}")
        conn.rollback()
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_batch_scoring()
