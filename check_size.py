#!/usr/bin/env python3
"""Check project size and identify large directories"""

import os
from pathlib import Path

def get_size_mb(path):
    """Get size of file or directory in MB"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except:
            pass
        return total / (1024 * 1024)
    return 0

def main():
    project_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Project Size Analysis")
    print("=" * 60)
    
    items = []
    for item in project_dir.iterdir():
        if item.name.startswith('.'):
            continue
        size = get_size_mb(item)
        items.append((item.name, size))
    
    # Sort by size
    items.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 largest items:")
    total = 0
    for name, size in items[:10]:
        print(f"  {name:30s} {size:8.1f} MB")
        total += size
    
    print(f"\n{'Total (top 10)':30s} {total:8.1f} MB")
    
    # Check specific problematic directories
    print("\n" + "=" * 60)
    print("Problematic Directories (should be excluded from zip):")
    print("=" * 60)
    
    problematic = ['venv', 'saved_models', '.git', '__pycache__']
    for name in problematic:
        path = project_dir / name
        if path.exists():
            size = get_size_mb(path)
            print(f"  {name:30s} {size:8.1f} MB")
    
    print("\n" + "=" * 60)
    print("Recommendation:")
    print("=" * 60)
    print("Exclude these from zip:")
    print("  - venv/ (virtual environment - users create their own)")
    print("  - saved_models/ (model files - too large, users train their own)")
    print("  - .git/ (git history - not needed for submission)")
    print("  - __pycache__/ (Python cache - auto-generated)")

if __name__ == '__main__':
    main()

