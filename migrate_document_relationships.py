#!/usr/bin/env python
"""
Migration script to add document_relationships table to the database.

This script adds the necessary table and indices to track relationships
between original PDFs and derived documents in the preprocessing pipeline.
"""

import os
import sys
import logging
import argparse
from typing import Optional

from claimctl.database import get_db_connection
from claimctl.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_document_relationships_table(conn) -> None:
    """
    Create the document_relationships table and indices.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    logger.info("Creating document_relationships table...")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        matter_id TEXT NOT NULL,
        original_pdf_path TEXT NOT NULL,
        derived_pdf_path TEXT NOT NULL,
        original_pdf_hash TEXT,
        derived_pdf_hash TEXT,
        relationship_type TEXT NOT NULL,
        start_page INTEGER,
        end_page INTEGER,
        document_type TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    logger.info("Creating indices...")
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_doc_rel_original 
    ON document_relationships(original_pdf_path, matter_id)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_doc_rel_derived 
    ON document_relationships(derived_pdf_path, matter_id)
    """)
    
    conn.commit()
    logger.info("Document relationships table created successfully")

def check_table_exists(conn, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        conn: Database connection
        table_name: Name of the table to check
        
    Returns:
        True if the table exists, False otherwise
    """
    cursor = conn.cursor()
    cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name=?
    """, (table_name,))
    
    return cursor.fetchone() is not None

def main(args):
    # Load configuration
    config = load_config()
    
    # Get matter_id if specified
    matter_id = args.matter
    
    # Get database connection
    if matter_id:
        # Get matter-specific database
        from claimctl.cli import get_matter_by_name
        matter = get_matter_by_name(matter_id)
        if not matter:
            logger.error(f"Matter '{matter_id}' not found")
            return 1
        
        db_path = os.path.join(config.get('MATTER_DIR', 'matters'), matter.name, 'data', 'documents.db')
        logger.info(f"Using matter-specific database: {db_path}")
        conn = get_db_connection(db_path)
    else:
        # Use default database
        logger.info("Using default database")
        conn = get_db_connection()
    
    try:
        # Check if table already exists
        if check_table_exists(conn, 'document_relationships'):
            if args.force:
                logger.info("Table 'document_relationships' already exists, dropping and recreating...")
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS document_relationships")
                conn.commit()
                create_document_relationships_table(conn)
            else:
                logger.info("Table 'document_relationships' already exists, skipping creation")
        else:
            create_document_relationships_table(conn)
        
        logger.info("Migration completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return 1
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate database to add document relationships table')
    parser.add_argument('-m', '--matter', help='Name of the matter to update')
    parser.add_argument('-f', '--force', action='store_true', help='Force recreation of table if it exists')
    args = parser.parse_args()
    
    sys.exit(main(args))