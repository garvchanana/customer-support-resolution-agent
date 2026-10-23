from app.db import test_database_connection

try:
    version = test_database_connection()
    print("Database connection successful!")
    print(version)
except Exception as exc:
    print("Database connection failed.")
    print(f"Error type: {type(exc).__name__}")
    print(exc)
