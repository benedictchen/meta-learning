#!/usr/bin/env python3
"""
Function Inventory Extractor for Meta-Learning Package
=====================================================

Extracts all functions, classes, and methods from the pre-v3 implementation
to ensure nothing valuable is lost in the v2 rewrite.
"""

import subprocess
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

class FunctionInventory:
    """Extract and catalog all functions from git history."""
    
    def __init__(self, commit_hash: str = "3e82159^"):
        self.commit_hash = commit_hash
        self.inventory = defaultdict(list)
        
    def get_all_python_files(self) -> List[str]:
        """Get all Python files from the specified commit."""
        result = subprocess.run([
            "git", "ls-tree", "-r", "--name-only", self.commit_hash
        ], capture_output=True, text=True, cwd=".")
        
        files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
        return files
    
    def extract_functions_from_file(self, filepath: str) -> Dict[str, List[str]]:
        """Extract all function/class definitions from a file."""
        try:
            result = subprocess.run([
                "git", "show", f"{self.commit_hash}:{filepath}"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode != 0:
                return {"classes": [], "functions": [], "methods": []}
            
            content = result.stdout
            
            # Extract classes
            class_pattern = r'^class\s+(\w+).*?:'
            classes = re.findall(class_pattern, content, re.MULTILINE)
            
            # Extract functions (not methods)
            func_pattern = r'^def\s+(\w+)\s*\('
            functions = []
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                # Check if it's inside a class (rough heuristic)
                line_start = content[:match.start()].count('\n')
                lines_before = content.split('\n')[:line_start]
                
                # Look for class definition in recent lines
                is_method = False
                for i in range(min(50, len(lines_before))):
                    line = lines_before[-(i+1)].strip()
                    if line.startswith('class '):
                        is_method = True
                        break
                    elif line and not line.startswith(' ') and not line.startswith('#'):
                        break
                
                if is_method:
                    continue
                functions.append(match.group(1))
            
            # Extract methods (functions inside classes)
            methods = []
            class_blocks = re.finditer(r'^class\s+\w+.*?:', content, re.MULTILINE)
            for class_match in class_blocks:
                class_start = class_match.end()
                # Find method definitions within the class
                remaining_content = content[class_start:]
                method_matches = re.finditer(r'^\s+def\s+(\w+)\s*\(', remaining_content, re.MULTILINE)
                for method_match in method_matches:
                    methods.append(method_match.group(1))
            
            return {
                "classes": classes,
                "functions": functions,
                "methods": methods
            }
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return {"classes": [], "functions": [], "methods": []}
    
    def build_inventory(self) -> None:
        """Build complete function inventory."""
        files = self.get_all_python_files()
        
        for filepath in files:
            if not filepath:  # Skip empty strings
                continue
                
            print(f"Processing: {filepath}")
            file_functions = self.extract_functions_from_file(filepath)
            
            self.inventory[filepath] = {
                "classes": file_functions["classes"],
                "functions": file_functions["functions"],
                "methods": file_functions["methods"],
                "total_items": len(file_functions["classes"]) + len(file_functions["functions"]) + len(file_functions["methods"])
            }
    
    def analyze_critical_modules(self) -> Dict[str, any]:
        """Identify the most critical modules by function count and type."""
        critical_files = {}
        
        # Key algorithm files to prioritize
        algorithm_patterns = [
            "test_time_compute",
            "maml",
            "research_patches",
            "few_shot",
            "batch_norm",
            "determinism"
        ]
        
        for filepath, data in self.inventory.items():
            if data["total_items"] > 5:  # Files with significant functionality
                critical_files[filepath] = data
                
            # Mark algorithm files as critical regardless of size
            for pattern in algorithm_patterns:
                if pattern in filepath:
                    critical_files[filepath] = data
                    break
        
        return critical_files
    
    def generate_report(self) -> str:
        """Generate a comprehensive function inventory report."""
        total_classes = sum(len(data["classes"]) for data in self.inventory.values())
        total_functions = sum(len(data["functions"]) for data in self.inventory.values())
        total_methods = sum(len(data["methods"]) for data in self.inventory.values())
        
        critical_modules = self.analyze_critical_modules()
        
        report = f"""
META-LEARNING FUNCTION INVENTORY REPORT
======================================
Commit: {self.commit_hash}
Total Files Analyzed: {len(self.inventory)}

SUMMARY STATISTICS:
- Total Classes: {total_classes}
- Total Functions: {total_functions}  
- Total Methods: {total_methods}
- Total Items: {total_classes + total_functions + total_methods}

CRITICAL MODULES ({len(critical_modules)} files):
"""
        
        for filepath, data in sorted(critical_modules.items(), key=lambda x: x[1]["total_items"], reverse=True):
            report += f"\n📄 {filepath}"
            report += f"\n   Classes: {len(data['classes'])} | Functions: {len(data['functions'])} | Methods: {len(data['methods'])}"
            if data['classes']:
                report += f"\n   🏗️  Classes: {', '.join(data['classes'][:5])}{'...' if len(data['classes']) > 5 else ''}"
            if data['functions']:
                report += f"\n   ⚡ Functions: {', '.join(data['functions'][:5])}{'...' if len(data['functions']) > 5 else ''}"
            report += "\n"
        
        return report
    
    def save_inventory(self, filepath: str = "meta_learning_function_inventory.json"):
        """Save complete inventory to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(dict(self.inventory), f, indent=2)
        print(f"Inventory saved to: {filepath}")

if __name__ == "__main__":
    print("🔍 Building Meta-Learning Function Inventory...")
    inventory = FunctionInventory()
    inventory.build_inventory()
    
    print("\n📊 Generating Report...")
    report = inventory.generate_report()
    print(report)
    
    print("\n💾 Saving Inventory...")
    inventory.save_inventory()
    
    print("\n✅ Function Inventory Complete!")