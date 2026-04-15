import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import bcrypt
import pandas as pd
import requests
from sqlalchemy import create_engine, text, bindparam

from .utils import get_database_url

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def update_last_login(user_id: int) -> None:
    """
    Update the last login timestamp for a user.

    Parameters:
        user_id (int): User's ID
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        UPDATE quant__euler_vaults_dashboard__users 
        SET last_login = CURRENT_TIMESTAMP 
        WHERE _id = :user_id
        """

        with engine.connect() as conn:
            conn.execute(text(query), {"user_id": user_id})
            conn.commit()

    except Exception as e:
        logging.error("Error updating last login: %s", str(e))
    finally:
        if "engine" in locals():
            engine.dispose()


def login_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Authenticate a user with email and password.

    Parameters:
        email (str): User's email address
        password (str): User's password

    Returns:
        Tuple[bool, str, Optional[Dict]]: (success, message, user_data)
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        SELECT _id, email, password_hash, first_name, last_name, is_active, is_admin, last_login
        FROM quant__euler_vaults_dashboard__users 
        WHERE email = :email
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"email": email.lower()})
            user = result.fetchone()

        if not user:
            return False, "Invalid email or password", None

        # Check if user is active
        if not user.is_active:
            return False, "Account is deactivated", None

        # Verify password
        stored_hash = user.password_hash.encode("utf-8")
        if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
            return False, "Invalid email or password", None

        # Update last login
        update_last_login(user._id)

        # Return user data (excluding password hash)
        user_data = {
            "id": user._id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_admin": user.is_admin,
            "last_login": user.last_login,
        }

        logging.info("Successful login for user: %s", email)
        return True, "Login successful", user_data

    except Exception as e:
        logging.error("Error during login: %s", str(e))
        return False, f"Login failed: {str(e)}", None
    finally:
        if "engine" in locals():
            engine.dispose()


def is_email_whitelisted(email: str) -> Tuple[bool, bool]:
    """
    Check if an email is whitelisted for registration and return additional info.

    Parameters:
        email (str): Email address to check

    Returns:
        Tuple[bool, bool]: (is_whitelisted, is_admin)
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        SELECT is_admin
        FROM quant__euler_vaults_dashboard__email_whitelist 
        WHERE email = :email AND is_active = TRUE
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"email": email.lower()})
            row = result.fetchone()

        if row:
            is_admin = row[0] if row[0] is not None else False
            return True, is_admin
        else:
            return False, False

    except Exception as e:
        logging.error("Error checking email whitelist: %s", str(e))
        return False, False
    finally:
        if "engine" in locals():
            engine.dispose()


def user_exists(email: str) -> bool:
    """
    Check if a user with the given email already exists.

    Parameters:
        email (str): Email address to check

    Returns:
        bool: True if user exists, False otherwise
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        SELECT COUNT(*) as count 
        FROM quant__euler_vaults_dashboard__users 
        WHERE email = :email
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"email": email.lower()})
            count = result.fetchone()[0]

        return count > 0

    except Exception as e:
        logging.error("Error checking if user exists: %s", str(e))
        return False
    finally:
        if "engine" in locals():
            engine.dispose()


def register_user(
    email: str, password: str, first_name: str = None, last_name: str = None
) -> Tuple[bool, str]:
    """
    Register a new user with secure password hashing.

    Parameters:
        email (str): User's email address
        password (str): User's password (will be hashed)
        first_name (str, optional): User's first name
        last_name (str, optional): User's last name

    Returns:
        Tuple[bool, str]: (success, message)
    """
    database_url = get_database_url()

    try:
        # Check if email is whitelisted
        is_whitelisted, is_admin = is_email_whitelisted(email)
        if not is_whitelisted:
            return False, "Email is not whitelisted for registration"

        # Check if user already exists
        if user_exists(email):
            return False, "User with this email already exists"

        # Hash the password
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        engine = create_engine(database_url)

        query = """
        INSERT INTO quant__euler_vaults_dashboard__users (email, password_hash, first_name, last_name, is_admin)
        VALUES (:email, :password_hash, :first_name, :last_name, :is_admin)
        """

        with engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    "email": email.lower(),
                    "password_hash": password_hash.decode("utf-8"),
                    "first_name": first_name,
                    "last_name": last_name,
                    "is_admin": is_admin,
                },
            )
            conn.commit()

        logging.info("Successfully registered user: %s", email)
        return True, "User registered successfully"

    except Exception as e:
        logging.error("Error registering user: %s", str(e))
        return False, f"Registration failed: {str(e)}"
    finally:
        if "engine" in locals():
            engine.dispose()


def get_whitelisted_emails() -> pd.DataFrame:
    """
    Get all whitelisted emails.

    Returns:
        pd.DataFrame: DataFrame containing whitelisted emails
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        SELECT email, is_active, created_at, created_by, is_admin
        FROM quant__euler_vaults_dashboard__email_whitelist
        ORDER BY created_at DESC
        """

        df = pd.read_sql(query, engine)
        logging.info("Retrieved %d whitelisted emails", len(df))
        return df

    except Exception as e:
        logging.error("Error getting whitelisted emails: %s", str(e))
        return pd.DataFrame()
    finally:
        if "engine" in locals():
            engine.dispose()


def add_email_to_whitelist(
    email: str,
    created_by: str = None,
    is_admin: bool = False,
) -> Tuple[bool, str]:
    """
    Add an email to the whitelist for registration.

    Parameters:
        email (str): Email address to whitelist
        created_by (str, optional): Who added this email
        is_admin (bool, optional): Whether the user is an admin

    Returns:
        Tuple[bool, str]: (success, message)
    """
    database_url = get_database_url()

    try:

        engine = create_engine(database_url)

        query = """
        INSERT INTO quant__euler_vaults_dashboard__email_whitelist (email, created_by, is_admin)
        VALUES (:email, :created_by, :is_admin)
        ON CONFLICT (email) DO UPDATE SET
            is_active = TRUE,
            created_by = EXCLUDED.created_by,
            is_admin = EXCLUDED.is_admin
        """

        with engine.connect() as conn:
            conn.execute(
                text(query),
                {
                    "email": email.lower(),
                    "created_by": created_by,
                    "is_admin": is_admin,
                },
            )
            conn.commit()

        logging.info("Successfully added email to whitelist: %s", email)
        return True, "Email added to whitelist successfully"

    except Exception as e:
        logging.error("Error adding email to whitelist: %s", str(e))
        return False, f"Failed to add email to whitelist: {str(e)}"
    finally:
        if "engine" in locals():
            engine.dispose()


def remove_email_from_whitelist(email: str) -> Tuple[bool, str]:
    """
    Remove an email from the whitelist (deactivate it).

    Parameters:
        email (str): Email address to remove from whitelist

    Returns:
        Tuple[bool, str]: (success, message)
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        UPDATE quant__euler_vaults_dashboard__email_whitelist 
        SET is_active = FALSE 
        WHERE email = :email
        """

        with engine.connect() as conn:
            result = conn.execute(text(query), {"email": email.lower()})
            conn.commit()

            if result.rowcount == 0:
                return False, "Email not found in whitelist"

        logging.info("Successfully removed email from whitelist: %s", email)
        return True, "Email removed from whitelist successfully"

    except Exception as e:
        logging.error("Error removing email from whitelist: %s", str(e))
        return False, f"Failed to remove email from whitelist: {str(e)}"
    finally:
        if "engine" in locals():
            engine.dispose()


def remove_user(email: str) -> Tuple[bool, str]:
    """
    Remove a user from the database.
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        DELETE FROM quant__euler_vaults_dashboard__users 
        WHERE email = :email
        """

        with engine.connect() as conn:
            conn.execute(text(query), {"email": email.lower()})
            conn.commit()

        return True, "User removed successfully"

    except Exception as e:
        logging.error("Error removing user: %s", str(e))
        return False, f"Failed to remove user: {str(e)}"
    finally:
        if "engine" in locals():
            engine.dispose()


def get_users() -> pd.DataFrame:
    """
    Get all users (excluding password hashes).

    Returns:
        pd.DataFrame: DataFrame containing user information
    """
    database_url = get_database_url()

    try:
        engine = create_engine(database_url)

        query = """
        SELECT _id, email, first_name, last_name, is_active, is_admin, last_login, created_at, updated_at
        FROM quant__euler_vaults_dashboard__users
        ORDER BY created_at DESC
        """

        df = pd.read_sql(query, engine)
        logging.info("Retrieved %d users", len(df))
        return df

    except Exception as e:
        logging.error("Error getting users: %s", str(e))
        return pd.DataFrame()
    finally:
        if "engine" in locals():
            engine.dispose()