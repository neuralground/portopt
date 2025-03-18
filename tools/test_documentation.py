#!/usr/bin/env python
"""
Documentation test script for Portfolio Optimization Testbed.

This script tests the code examples in the documentation to ensure they are accurate and up-to-date.
It extracts Python code blocks from markdown files and runs them to verify they execute without errors.
"""

import os
import re
import sys
import doctest
import unittest
import subprocess
import tempfile
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"


class DocumentationTester:
    """Class for testing documentation code examples."""

    def __init__(self, docs_dir=DOCS_DIR):
        self.docs_dir = Path(docs_dir)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.results = {
            "tested_files": 0,
            "code_blocks": 0,
            "successful_blocks": 0,
            "failed_blocks": 0,
            "skipped_blocks": 0,
            "failures": []
        }

    def extract_python_blocks(self, markdown_content):
        """Extract Python code blocks from markdown content."""
        # Pattern to match Python code blocks: ```python ... ```
        pattern = r"```python\n(.*?)```"
        return re.findall(pattern, markdown_content, re.DOTALL)

    def test_code_block(self, code_block, file_path, block_index):
        """Test a single code block by executing it."""
        # Skip certain blocks that are not meant to be run directly
        if "import " not in code_block and "def " not in code_block and "class " not in code_block:
            self.results["skipped_blocks"] += 1
            return True

        # Skip blocks with placeholders or incomplete code
        if "<...>" in code_block or "..." in code_block:
            self.results["skipped_blocks"] += 1
            return True

        # Create a temporary Python file
        temp_file = self.temp_dir / f"test_block_{block_index}.py"
        with open(temp_file, "w") as f:
            f.write(code_block)

        # Execute the code block
        try:
            subprocess.run(
                [sys.executable, str(temp_file)],
                check=True,
                capture_output=True,
                text=True
            )
            self.results["successful_blocks"] += 1
            return True
        except subprocess.CalledProcessError as e:
            self.results["failed_blocks"] += 1
            self.results["failures"].append({
                "file": str(file_path),
                "block_index": block_index,
                "code": code_block,
                "error": e.stderr
            })
            return False

    def test_markdown_file(self, file_path):
        """Test all Python code blocks in a markdown file."""
        with open(file_path, "r") as f:
            content = f.read()

        code_blocks = self.extract_python_blocks(content)
        self.results["code_blocks"] += len(code_blocks)

        for i, block in enumerate(code_blocks):
            self.test_code_block(block, file_path, i)

        self.results["tested_files"] += 1

    def test_all_markdown_files(self):
        """Test all markdown files in the documentation directory."""
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = Path(root) / file
                    self.test_markdown_file(file_path)

    def test_doctests(self):
        """Run doctests on all Python files in the project."""
        project_dir = self.docs_dir.parent / "portopt"
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    try:
                        doctest.testfile(
                            str(file_path),
                            module_relative=False,
                            verbose=False
                        )
                    except doctest.DocTestFailure as e:
                        self.results["failures"].append({
                            "file": str(file_path),
                            "block_index": "doctest",
                            "code": e.example.source,
                            "error": f"Expected: {e.example.want}, Got: {e.got}"
                        })

    def test_notebooks(self):
        """Test Jupyter notebooks using nbconvert."""
        notebooks_dir = self.docs_dir / "examples" / "notebooks"
        if not notebooks_dir.exists():
            return

        for notebook in notebooks_dir.glob("*.ipynb"):
            try:
                subprocess.run(
                    [
                        "jupyter", "nbconvert", "--to", "notebook", 
                        "--execute", str(notebook), "--output", str(notebook.name)
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.results["successful_blocks"] += 1
            except subprocess.CalledProcessError as e:
                self.results["failed_blocks"] += 1
                self.results["failures"].append({
                    "file": str(notebook),
                    "block_index": "notebook",
                    "code": "Entire notebook",
                    "error": e.stderr
                })

    def print_results(self):
        """Print the test results."""
        print("\n=== Documentation Test Results ===")
        print(f"Tested files: {self.results['tested_files']}")
        print(f"Code blocks: {self.results['code_blocks']}")
        print(f"Successful blocks: {self.results['successful_blocks']}")
        print(f"Failed blocks: {self.results['failed_blocks']}")
        print(f"Skipped blocks: {self.results['skipped_blocks']}")
        
        if self.results["failures"]:
            print("\n=== Failures ===")
            for i, failure in enumerate(self.results["failures"]):
                print(f"\nFailure {i+1}:")
                print(f"File: {failure['file']}")
                print(f"Block: {failure['block_index']}")
                print("Code:")
                print("```")
                print(failure["code"])
                print("```")
                print("Error:")
                print(failure["error"])
        
        success_rate = (
            self.results["successful_blocks"] / 
            (self.results["successful_blocks"] + self.results["failed_blocks"])
            * 100 if (self.results["successful_blocks"] + self.results["failed_blocks"]) > 0 else 0
        )
        print(f"\nSuccess rate: {success_rate:.2f}%")
        
        return self.results["failed_blocks"] == 0


def main():
    """Main function to run the documentation tests."""
    tester = DocumentationTester()
    print("Testing markdown code blocks...")
    tester.test_all_markdown_files()
    
    print("Testing doctests...")
    tester.test_doctests()
    
    print("Testing notebooks...")
    tester.test_notebooks()
    
    success = tester.print_results()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
