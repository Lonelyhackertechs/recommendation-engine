import os

from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()


def get_connection():

    database_url = os.getenv(
        "DATABASE_URL"
    )

    return create_engine(
        database_url
    )
