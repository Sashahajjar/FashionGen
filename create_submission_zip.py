#!/usr/bin/env python3
"""
Create a clean submission zip file excluding large/unnecessary files
"""

import os
import zipfile
from pathlib import Path

# Directories and files to exclude
EXCLUDE_PATTERNS = [
    'venv',
    'saved_models',
    '.git',
    '__pycache__',
    '.DS_Store',
    '*.pyc',
    '*.pyo',
    '*.log',
    'data/images',
    'data/captions',
    'data/processed',
    '*.zip',
    '.ipynb_checkpoints',
]

def should_exclude(path_str):
    """Check if a path should be excluded"""
    path = Path(path_str)
    
    # Check exact matches
    if path.name in ['venv', 'saved_models', '.git', '__pycache__', '.DS_Store']:
        return True
    
    # Check if in excluded directories
    for part in path.parts:
        if part in ['venv', 'saved_models', '.git', '__pycache__', 'data']:
            # Special handling for data subdirectories
            if part == 'data' and path.parts[-1] in ['images', 'captions', 'processed']:
                return True
            if part in ['venv', 'saved_models', '.git', '__pycache__']:
                return True
    
    # Check extensions
    if path.suffix in ['.pyc', '.pyo', '.log', '.zip']:
        return True
    
    return False

def create_submission_zip():
    """Create a clean zip file for submission"""
    project_dir = Path(__file__).parent
    zip_name = project_dir / 'flickr8k-project-submission.zip'
    
    # Remove old zip if exists
    if zip_name.exists():
        zip_name.unlink()
        print(f"Removed old zip file: {zip_name.name}")
    
    print("Creating submission zip file...")
    print("Excluding: venv/, saved_models/, .git/, __pycache__/, data files")
    print()
    
    total_size = 0
    included_files = 0
    excluded_files = 0
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            # Remove excluded directories from dirs list to prevent walking into them
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(project_dir)
                
                # Skip the zip file itself
                if file_path == zip_name:
                    continue
                
                if should_exclude(str(rel_path)):
                    excluded_files += 1
                    continue
                
                try:
                    zipf.write(file_path, rel_path)
                    size = file_path.stat().st_size
                    total_size += size
                    included_files += 1
                except Exception as e:
                    print(f"Warning: Could not add {rel_path}: {e}")
    
    zip_size_mb = zip_name.stat().st_size / (1024 * 1024)
    
    print("=" * 60)
    print("Zip file created successfully!")
    print("=" * 60)
    print(f"File: {zip_name.name}")
    print(f"Size: {zip_size_mb:.2f} MB")
    print(f"Files included: {included_files}")
    print(f"Files excluded: {excluded_files}")
    print()
    print("Excluded directories:")
    print("  - venv/ (virtual environment)")
    print("  - saved_models/ (trained models)")
    print("  - .git/ (git history)")
    print("  - __pycache__/ (Python cache)")
    print("  - data/images/, data/captions/, data/processed/ (data files)")

if __name__ == '__main__':
    create_submission_zip()

