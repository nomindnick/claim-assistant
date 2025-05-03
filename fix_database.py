#!/usr/bin/env python
"""Script to fix database schema."""

import os
import shutil
from pathlib import Path

from claimctl.config import get_config
from claimctl.database import get_database_engine, Base, init_database

def main():
    """Fix database schema by recreating it."""
    # Get database path
    config = get_config()
    db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
    
    print(f"Current database path: {db_path}")
    
    # Backup the original database
    if db_path.exists():
        backup_path = db_path.with_suffix(".db.bak")
        print(f"Backing up database to: {backup_path}")
        shutil.copy2(db_path, backup_path)
        
        # Remove the old database file
        print("Removing old database file...")
        os.remove(db_path)
    
    # Initialize fresh database with correct schema
    print("Creating new database with correct schema...")
    
    # Get engine and recreate all tables from scratch
    engine = get_database_engine()
    Base.metadata.drop_all(engine)  # Drop all tables to ensure clean slate
    Base.metadata.create_all(engine)  # Create all tables with current schema
    
    print("Database schema has been fixed successfully!")
    print(f"If you need to restore your data, the backup is at: {backup_path}")
    
    # Verify the fix worked
    try:
        import sqlalchemy as sa
        inspector = sa.inspect(engine)
        doc_columns = [col['name'] for col in inspector.get_columns('documents')]
        if 'matter_id' in doc_columns:
            print("Verification successful: 'matter_id' column is now present in documents table.")
        else:
            print("WARNING: 'matter_id' column is still missing! This is unexpected.")
    except Exception as e:
        print(f"Error verifying fix: {str(e)}")

if __name__ == "__main__":
    main()