#!/usr/bin/env python
"""
Script to update API reference links in documentation.

This script finds all references to the old API reference location and updates them to point to the new location.
"""

import os
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
OLD_PATH = "../user-guides/api-reference.md"
NEW_PATH = "../reference/api-reference.md"
OLD_PATH_VARIANTS = [
    "../user-guides/api-reference.md",
    "./api-reference.md",
    "./user-guides/api-reference.md"
]

def update_links_in_file(file_path):
    """Update API reference links in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    original_content = content
    
    # Replace links with various patterns
    for old_path in OLD_PATH_VARIANTS:
        # Replace links with and without anchors
        content = re.sub(
            r'\[([^\]]+)\]\(' + re.escape(old_path) + r'(#[^)]+)?\)',
            r'[\1](' + NEW_PATH + r'\2)',
            content
        )
    
    # Only write to the file if changes were made
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated links in {file_path}")
        return True
    
    return False

def update_all_files():
    """Update API reference links in all markdown files."""
    updated_files = 0
    
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                if update_links_in_file(file_path):
                    updated_files += 1
    
    return updated_files

if __name__ == "__main__":
    print("Updating API reference links...")
    updated_files = update_all_files()
    print(f"Updated links in {updated_files} files.")
