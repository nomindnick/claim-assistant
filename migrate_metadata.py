#!/usr/bin/env python3
"""Migration script to add new metadata columns to the page_chunks table."""

import sqlalchemy as sa
from claimctl.database import get_database_engine, get_session, PageChunk

def add_metadata_columns():
    """Add new metadata columns to the page_chunks table."""
    print("Adding new metadata columns to page_chunks table...")
    
    engine = get_database_engine()
    conn = engine.connect()
    
    # Check if columns exist
    inspector = sa.inspect(engine)
    existing_columns = [col['name'] for col in inspector.get_columns('page_chunks')]
    
    # Add columns if they don't exist
    for column_name in ['amount', 'time_period', 'section_reference', 
                       'public_agency_reference', 'work_description']:
        if column_name not in existing_columns:
            print(f"Adding column: {column_name}")
            conn.execute(sa.text(f"ALTER TABLE page_chunks ADD COLUMN {column_name} TEXT"))
        else:
            print(f"Column {column_name} already exists, skipping")
    
    # Add new indices if they don't exist
    existing_indices = [idx['name'] for idx in inspector.get_indexes('page_chunks')]
    
    if 'idx_section_reference' not in existing_indices:
        print("Adding index: idx_section_reference")
        conn.execute(sa.text("CREATE INDEX idx_section_reference ON page_chunks (section_reference)"))
    
    if 'idx_public_agency_reference' not in existing_indices:
        print("Adding index: idx_public_agency_reference")
        conn.execute(sa.text("CREATE INDEX idx_public_agency_reference ON page_chunks (public_agency_reference)"))
    
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    add_metadata_columns()