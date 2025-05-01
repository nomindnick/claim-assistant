#!/usr/bin/env python
"""Script to diagnose database issues."""

from claimctl.database import get_database_engine, Base
import sqlalchemy as sa

def main():
    """Diagnose database schema."""
    engine = get_database_engine()
    inspector = sa.inspect(engine)
    
    print('Available tables:', inspector.get_table_names())
    
    if 'documents' in inspector.get_table_names():
        print('Document columns:', [col['name'] for col in inspector.get_columns('documents')])
    
    if 'matters' in inspector.get_table_names():
        print('Matter columns:', [col['name'] for col in inspector.get_columns('matters')])
    
    # Check if we need to add the matter_id column
    if 'documents' in inspector.get_table_names():
        doc_columns = [col['name'] for col in inspector.get_columns('documents')]
        if 'matter_id' not in doc_columns:
            print("WARNING: 'matter_id' column is missing from documents table!")
            print("Database structure needs to be updated.")

if __name__ == "__main__":
    main()