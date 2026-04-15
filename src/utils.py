import os

from dotenv import load_dotenv

load_dotenv()


def get_database_url():
    """
    Get database URL from environment variables.

    Returns:
        str: Database connection URL
    """
    # You can customize these environment variable names as needed
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "itb_quant_database")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
