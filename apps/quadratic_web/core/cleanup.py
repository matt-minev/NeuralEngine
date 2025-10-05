#!/usr/bin/env python3
"""
Dataset Cleanup Script
Removes uploaded datasets older than specified time
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_old_datasets(uploads_folder='uploads', max_age_days=1, dry_run=False):
    """
    Clean up dataset files older than specified days
    
    Args:
        uploads_folder (str): Path to uploads folder
        max_age_days (int): Maximum age in days before deletion
        dry_run (bool): If True, only show what would be deleted
    """
    if not os.path.exists(uploads_folder):
        logger.warning(f"Uploads folder '{uploads_folder}' does not exist")
        return 0, 0

    # Calculate cutoff time
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    cutoff_date = datetime.fromtimestamp(cutoff_time)
    
    logger.info(f"Cleaning up files older than {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    deleted_count = 0
    total_size = 0
    files_checked = 0
    
    # Scan uploads folder
    for filename in os.listdir(uploads_folder):
        filepath = os.path.join(uploads_folder, filename)
        
        # Skip directories and non-CSV files
        if not os.path.isfile(filepath):
            continue
            
        # Check if it's a dataset file (CSV or similar)
        if not (filename.endswith('.csv') or filename.startswith('dataset_') or filename.startswith('quadratic_dataset_')):
            continue
            
        files_checked += 1
        
        # Get file modification time
        file_mtime = os.path.getmtime(filepath)
        
        if file_mtime < cutoff_time:
            file_size = os.path.getsize(filepath)
            file_date = datetime.fromtimestamp(file_mtime)
            
            if dry_run:
                logger.info(f"Would delete: {filename} (size: {file_size} bytes, modified: {file_date.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted: {filename} (size: {file_size} bytes, modified: {file_date.strftime('%Y-%m-%d %H:%M:%S')})")
                    deleted_count += 1
                    total_size += file_size
                except OSError as e:
                    logger.error(f"Failed to delete {filename}: {e}")
    
    # Summary
    if dry_run:
        logger.info(f"Dry run completed. Found {files_checked} dataset files")
    else:
        logger.info(f"Cleanup completed. Deleted {deleted_count} files, freed {total_size} bytes from {files_checked} dataset files")
    
    return deleted_count, total_size

def main():
    """Command line interface for cleanup script"""
    parser = argparse.ArgumentParser(description='Clean up old dataset files')
    parser.add_argument('--folder', default='uploads', help='Uploads folder path (default: uploads)')
    parser.add_argument('--days', type=int, default=1, help='Maximum age in days (default: 1)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    logger.info("🧹 Starting dataset cleanup...")
    logger.info(f"📁 Folder: {args.folder}")
    logger.info(f"📅 Max age: {args.days} days")
    logger.info(f"🔍 Dry run: {args.dry_run}")
    
    deleted_count, total_size = cleanup_old_datasets(
        uploads_folder=args.folder,
        max_age_days=args.days,
        dry_run=args.dry_run
    )
    
    if not args.dry_run and deleted_count > 0:
        logger.info(f"✅ Successfully cleaned up {deleted_count} files ({total_size:,} bytes)")
    elif not args.dry_run:
        logger.info("✅ No files needed cleanup")

if __name__ == '__main__':
    main()
