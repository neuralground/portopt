#!/usr/bin/env python
"""
Documentation link checker for Portfolio Optimization Testbed.

This script checks all links in the documentation to ensure they are valid.
It identifies broken internal links and creates placeholder files for missing documentation.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

DOCS_DIR = Path(__file__).parent.parent / "docs"


class LinkChecker:
    """Class for checking links in markdown documentation."""

    def __init__(self, docs_dir=DOCS_DIR):
        self.docs_dir = Path(docs_dir)
        self.all_files = set()
        self.broken_links = defaultdict(list)
        self.valid_links = defaultdict(list)
        self.placeholder_needed = defaultdict(list)

    def gather_all_files(self):
        """Gather all markdown files in the documentation directory."""
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".md") or file.endswith(".ipynb"):
                    rel_path = os.path.relpath(os.path.join(root, file), self.docs_dir)
                    self.all_files.add(rel_path)

    def extract_links(self, markdown_content):
        """Extract all markdown links from content."""
        # Pattern to match markdown links: [text](url)
        pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        return re.findall(pattern, markdown_content)

    def normalize_path(self, base_path, link_path):
        """Normalize a relative path based on the base path."""
        if link_path.startswith("http://") or link_path.startswith("https://"):
            return None  # External link, we don't check these
        
        if link_path.startswith("#"):
            return None  # Anchor link within the same page
        
        base_dir = os.path.dirname(base_path)
        if link_path.startswith("/"):
            # Absolute path within the docs
            normalized = link_path.lstrip("/")
        else:
            # Relative path
            normalized = os.path.normpath(os.path.join(base_dir, link_path))
        
        return normalized

    def check_file_links(self, file_path):
        """Check all links in a markdown file."""
        rel_path = os.path.relpath(file_path, self.docs_dir)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        links = self.extract_links(content)
        
        for text, link in links:
            normalized_link = self.normalize_path(rel_path, link)
            
            if normalized_link is None:
                continue  # External or anchor link
            
            if normalized_link in self.all_files:
                self.valid_links[rel_path].append((text, link, normalized_link))
            else:
                self.broken_links[rel_path].append((text, link, normalized_link))
                
                # Check if it's a missing documentation file that we should create
                if normalized_link.endswith(".md") and not os.path.exists(os.path.join(self.docs_dir, normalized_link)):
                    self.placeholder_needed[normalized_link].append((rel_path, text))

    def check_all_files(self):
        """Check links in all markdown files."""
        self.gather_all_files()
        
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    self.check_file_links(file_path)

    def create_placeholder(self, missing_file):
        """Create a placeholder file for a missing documentation link."""
        placeholder_path = os.path.join(self.docs_dir, missing_file)
        os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
        
        title = os.path.splitext(os.path.basename(missing_file))[0].replace("-", " ").title()
        
        with open(placeholder_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write("**This documentation is currently under development.**\n\n")
            f.write("We're working on creating comprehensive documentation for this topic. ")
            f.write("Check back soon for updates, or contribute to this documentation by submitting a pull request.\n\n")
            f.write("## Coming Soon\n\n")
            f.write("- Detailed explanation of concepts\n")
            f.write("- Code examples\n")
            f.write("- Best practices\n")
            f.write("- Related resources\n")
        
        print(f"Created placeholder: {missing_file}")
        return placeholder_path

    def create_all_placeholders(self):
        """Create placeholder files for all missing documentation links."""
        created_files = []
        for missing_file in self.placeholder_needed:
            created_files.append(self.create_placeholder(missing_file))
        return created_files

    def print_report(self):
        """Print a report of the link check results."""
        print("\n=== Documentation Link Check Report ===")
        print(f"Total files checked: {len(self.valid_links) + len(self.broken_links)}")
        print(f"Files with broken links: {len(self.broken_links)}")
        print(f"Total broken links: {sum(len(links) for links in self.broken_links.values())}")
        print(f"Missing files that need placeholders: {len(self.placeholder_needed)}")
        
        if self.broken_links:
            print("\n=== Broken Links ===")
            for file, links in self.broken_links.items():
                print(f"\nIn {file}:")
                for text, link, normalized in links:
                    print(f"  - [{text}]({link}) -> {normalized}")
        
        if self.placeholder_needed:
            print("\n=== Missing Documentation Files ===")
            for missing_file, references in self.placeholder_needed.items():
                print(f"\n{missing_file} is referenced from:")
                for ref_file, ref_text in references:
                    print(f"  - {ref_file}: [{ref_text}]")


def main():
    """Main function to run the link checker."""
    checker = LinkChecker()
    print("Checking documentation links...")
    checker.check_all_files()
    
    checker.print_report()
    
    # Automatically create placeholders without asking
    if checker.placeholder_needed:
        print("\nCreating placeholders for missing files...")
        created_files = checker.create_all_placeholders()
        print(f"\nCreated {len(created_files)} placeholder files.")
    
    return 0 if not checker.broken_links else 1


if __name__ == "__main__":
    sys.exit(main())
