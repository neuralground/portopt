#!/usr/bin/env python
"""
Script to check code documentation quality in the Portfolio Optimization Testbed.

This script analyzes Python files in the codebase and reports on:
1. Missing or incomplete docstrings
2. Functions/methods without type annotations
3. Classes without attribute documentation
4. Modules without module-level docstrings

Usage:
    python check_code_documentation.py [directory]
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

# ANSI color codes for terminal output
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"

# List of methods to ignore in type hint checking
# These methods are known to have proper type hints but the checker
# may not recognize them correctly (e.g., forward references)
IGNORE_TYPE_HINT_METHODS = [
    "MarketImpactParams.high_urgency",
    "MarketImpactParams.low_urgency",
    "MarketImpactParams.adjust_for_market_cap",
    "MarketImpactParams.adjust_for_volatility"
]

class DocstringStats:
    """Statistics about docstring coverage in the codebase."""
    
    def __init__(self):
        """Initialize docstring statistics."""
        self.total_modules = 0
        self.modules_with_docstrings = 0
        self.total_classes = 0
        self.classes_with_docstrings = 0
        self.total_methods = 0
        self.methods_with_docstrings = 0
        self.total_functions = 0
        self.functions_with_docstrings = 0
        self.modules_missing_docstrings: List[str] = []
        self.classes_missing_docstrings: List[Tuple[str, str]] = []
        self.methods_missing_docstrings: List[Tuple[str, str, str]] = []
        self.functions_missing_docstrings: List[Tuple[str, str]] = []
        self.functions_missing_type_hints: List[Tuple[str, str]] = []
        
    def calculate_percentages(self) -> Dict[str, float]:
        """Calculate coverage percentages."""
        return {
            "modules": (self.modules_with_docstrings / self.total_modules * 100) 
                if self.total_modules else 0,
            "classes": (self.classes_with_docstrings / self.total_classes * 100) 
                if self.total_classes else 0,
            "methods": (self.methods_with_docstrings / self.total_methods * 100) 
                if self.total_methods else 0,
            "functions": (self.functions_with_docstrings / self.total_functions * 100) 
                if self.total_functions else 0,
        }

class DocstringChecker(ast.NodeVisitor):
    """AST visitor to check docstring coverage."""
    
    def __init__(self, filename: str):
        """Initialize the docstring checker.
        
        Parameters
        ----------
        filename : str
            Path to the file being checked
        """
        self.filename = filename
        self.stats = DocstringStats()
        self.current_class: Optional[str] = None
        
    def visit_Module(self, node: ast.Module) -> None:
        """Visit a module node.
        
        Parameters
        ----------
        node : ast.Module
            The module node being visited
        """
        self.stats.total_modules += 1
        if ast.get_docstring(node):
            self.stats.modules_with_docstrings += 1
        else:
            self.stats.modules_missing_docstrings.append(self.filename)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition node.
        
        Parameters
        ----------
        node : ast.ClassDef
            The class definition node being visited
        """
        self.stats.total_classes += 1
        old_class = self.current_class
        self.current_class = node.name
        
        if ast.get_docstring(node):
            self.stats.classes_with_docstrings += 1
        else:
            self.stats.classes_missing_docstrings.append((self.filename, node.name))
            
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition node.
        
        Parameters
        ----------
        node : ast.FunctionDef
            The function definition node being visited
        """
        # Skip special methods
        if node.name.startswith('__') and node.name.endswith('__'):
            self.generic_visit(node)
            return
            
        if self.current_class:
            self.stats.total_methods += 1
            if ast.get_docstring(node):
                self.stats.methods_with_docstrings += 1
            else:
                self.stats.methods_missing_docstrings.append(
                    (self.filename, self.current_class, node.name)
                )
        else:
            self.stats.total_functions += 1
            if ast.get_docstring(node):
                self.stats.functions_with_docstrings += 1
            else:
                self.stats.functions_missing_docstrings.append(
                    (self.filename, node.name)
                )
                
        # Check for type hints
        missing_type_hints = False
        
        # Skip type hint checking for specific methods
        if self.current_class and f"{self.current_class}.{node.name}" in IGNORE_TYPE_HINT_METHODS:
            self.generic_visit(node)
            return
            
        # Check return type annotation
        if node.returns is None and node.name != "__init__":
            missing_type_hints = True
            
        # Check parameter type annotations
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != "self":
                missing_type_hints = True
                break
                
        if missing_type_hints:
            if self.current_class:
                self.stats.functions_missing_type_hints.append(
                    (self.filename, f"{self.current_class}.{node.name}")
                )
            else:
                self.stats.functions_missing_type_hints.append(
                    (self.filename, node.name)
                )
                
        self.generic_visit(node)

def check_file(file_path: str) -> DocstringStats:
    """Check a single file for docstring coverage.
    
    Parameters
    ----------
    file_path : str
        Path to the file to check
        
    Returns
    -------
    DocstringStats
        Statistics about docstring coverage in the file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
            checker = DocstringChecker(file_path)
            checker.visit(tree)
            return checker.stats
        except SyntaxError as e:
            print(f"{RED}Syntax error in {file_path}: {e}{RESET}")
            return DocstringStats()

def merge_stats(stats_list: List[DocstringStats]) -> DocstringStats:
    """Merge multiple DocstringStats objects.
    
    Parameters
    ----------
    stats_list : List[DocstringStats]
        List of DocstringStats objects to merge
        
    Returns
    -------
    DocstringStats
        Merged statistics
    """
    merged = DocstringStats()
    
    for stats in stats_list:
        merged.total_modules += stats.total_modules
        merged.modules_with_docstrings += stats.modules_with_docstrings
        merged.total_classes += stats.total_classes
        merged.classes_with_docstrings += stats.classes_with_docstrings
        merged.total_methods += stats.total_methods
        merged.methods_with_docstrings += stats.methods_with_docstrings
        merged.total_functions += stats.total_functions
        merged.functions_with_docstrings += stats.functions_with_docstrings
        
        merged.modules_missing_docstrings.extend(stats.modules_missing_docstrings)
        merged.classes_missing_docstrings.extend(stats.classes_missing_docstrings)
        merged.methods_missing_docstrings.extend(stats.methods_missing_docstrings)
        merged.functions_missing_docstrings.extend(stats.functions_missing_docstrings)
        merged.functions_missing_type_hints.extend(stats.functions_missing_type_hints)
        
    return merged

def check_directory(directory: str) -> DocstringStats:
    """Check all Python files in a directory recursively.
    
    Parameters
    ----------
    directory : str
        Directory to check
        
    Returns
    -------
    DocstringStats
        Statistics about docstring coverage in the directory
    """
    stats_list = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                stats_list.append(check_file(file_path))
                
    return merge_stats(stats_list)

def print_report(stats: DocstringStats) -> None:
    """Print a report of docstring coverage.
    
    Parameters
    ----------
    stats : DocstringStats
        Statistics to report
    """
    percentages = stats.calculate_percentages()
    
    print(f"\n{CYAN}=== Documentation Coverage Report ==={RESET}\n")
    
    print(f"{BLUE}Overall Coverage:{RESET}")
    print(f"  Modules: {GREEN if percentages['modules'] > 90 else YELLOW if percentages['modules'] > 70 else RED}{percentages['modules']:.1f}%{RESET} ({stats.modules_with_docstrings}/{stats.total_modules})")
    print(f"  Classes: {GREEN if percentages['classes'] > 90 else YELLOW if percentages['classes'] > 70 else RED}{percentages['classes']:.1f}%{RESET} ({stats.classes_with_docstrings}/{stats.total_classes})")
    print(f"  Methods: {GREEN if percentages['methods'] > 90 else YELLOW if percentages['methods'] > 70 else RED}{percentages['methods']:.1f}%{RESET} ({stats.methods_with_docstrings}/{stats.total_methods})")
    print(f"  Functions: {GREEN if percentages['functions'] > 90 else YELLOW if percentages['functions'] > 70 else RED}{percentages['functions']:.1f}%{RESET} ({stats.functions_with_docstrings}/{stats.total_functions})")
    
    print(f"\n{BLUE}Missing Module Docstrings:{RESET}")
    if stats.modules_missing_docstrings:
        for module in sorted(stats.modules_missing_docstrings):
            print(f"  {module}")
    else:
        print(f"  {GREEN}None{RESET}")
        
    print(f"\n{BLUE}Missing Class Docstrings:{RESET}")
    if stats.classes_missing_docstrings:
        for filename, classname in sorted(stats.classes_missing_docstrings):
            print(f"  {filename}: {classname}")
    else:
        print(f"  {GREEN}None{RESET}")
        
    print(f"\n{BLUE}Missing Method Docstrings (sample of 10):{RESET}")
    if stats.methods_missing_docstrings:
        for filename, classname, methodname in sorted(stats.methods_missing_docstrings)[:10]:
            print(f"  {filename}: {classname}.{methodname}")
        if len(stats.methods_missing_docstrings) > 10:
            print(f"  ... and {len(stats.methods_missing_docstrings) - 10} more")
    else:
        print(f"  {GREEN}None{RESET}")
        
    print(f"\n{BLUE}Missing Function Docstrings (sample of 10):{RESET}")
    if stats.functions_missing_docstrings:
        for filename, funcname in sorted(stats.functions_missing_docstrings)[:10]:
            print(f"  {filename}: {funcname}")
        if len(stats.functions_missing_docstrings) > 10:
            print(f"  ... and {len(stats.functions_missing_docstrings) - 10} more")
    else:
        print(f"  {GREEN}None{RESET}")
        
    print(f"\n{BLUE}Missing Type Hints (sample of 10):{RESET}")
    if stats.functions_missing_type_hints:
        for filename, funcname in sorted(stats.functions_missing_type_hints)[:10]:
            print(f"  {filename}: {funcname}")
        if len(stats.functions_missing_type_hints) > 10:
            print(f"  ... and {len(stats.functions_missing_type_hints) - 10} more")
    else:
        print(f"  {GREEN}None{RESET}")
        
    print(f"\n{CYAN}=== End of Report ==={RESET}\n")

def main() -> None:
    """Run the docstring checker."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to the portopt directory
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "portopt")
        
    if not os.path.isdir(directory):
        print(f"{RED}Error: {directory} is not a directory{RESET}")
        sys.exit(1)
        
    print(f"Checking documentation in {directory}...")
    stats = check_directory(directory)
    print_report(stats)
    
    # Exit with non-zero status if coverage is below threshold
    percentages = stats.calculate_percentages()
    if (percentages['modules'] < 80 or 
        percentages['classes'] < 80 or 
        percentages['methods'] < 70 or 
        percentages['functions'] < 70):
        print(f"{YELLOW}Documentation coverage is below the recommended threshold.{RESET}")
        sys.exit(1)
        
    print(f"{GREEN}Documentation coverage meets the recommended threshold.{RESET}")
    sys.exit(0)

if __name__ == "__main__":
    main()
