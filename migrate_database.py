#!/usr/bin/env python
"""Script to migrate database schema without losing data."""

import os
import shutil
from pathlib import Path
import sqlite3

from claimctl.config import get_config
from claimctl.database import get_database_engine, Base

def main():
    """Migrate database schema by altering existing tables."""
    # Get database path
    config = get_config()
    db_path = Path(config.paths.INDEX_DIR) / "catalog.db"
    
    print(f"Current database path: {db_path}")
    
    # Backup the original database
    if db_path.exists():
        backup_path = db_path.with_suffix(".db.bak")
        print(f"Backing up database to: {backup_path}")
        shutil.copy2(db_path, backup_path)
        
        # Connect to the SQLite database
        print("Connecting to database...")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if matter_id column exists
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'matter_id' not in columns:
            print("Adding matter_id column to documents table...")
            # Use ALTER TABLE to add the matter_id column 
            # SQLite doesn't support adding foreign key constraints in ALTER TABLE
            cursor.execute("ALTER TABLE documents ADD COLUMN matter_id INTEGER REFERENCES matters(id)")
            conn.commit()
            print("Column added successfully!")
        else:
            print("matter_id column already exists, no migration needed")
            
        # Verify that the column was added
        cursor.execute("PRAGMA table_info(documents)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'matter_id' in columns:
            print("Verification successful: matter_id column is present")
        else:
            print("ERROR: matter_id column is still missing!")
            
        conn.close()
    else:
        print(f"Database file does not exist at: {db_path}")
        print("No migration needed")

if __name__ == "__main__":
    main()