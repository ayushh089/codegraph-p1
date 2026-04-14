import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

class CodeParser:
    def __init__(self):
        self.functions = {}      # {function_name: file_path}
        self.classes = {}        # {class_name: file_path}
        self.calls = []          # [(caller, callee, file_path)]
        self.class_contains = [] # [(class_name, function_name, file_path)]
        
        # NEW: Track files and imports
        self.files = set()       # Set of all file paths
        self.imports = []        # [(from_file, to_module, line_number)]
        self.file_to_functions = []  # [(file_path, function_name)]
        self.file_to_classes = []    # [(file_path, class_name)]
        
    def parse_file(self, filepath: Path):
        """Parse a single Python file and extract structure"""
        try:
            source = filepath.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except Exception as e:
            print(f"  ⚠️  Error parsing {filepath}: {e}")
            return
        
        # Add file to set
        file_path_str = str(filepath)
        self.files.add(file_path_str)
        
        current_class = None
        
        for node in ast.walk(tree):
            # Extract functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                full_name = f"{func_name}"
                self.functions[full_name] = file_path_str
                
                # Track file contains function
                self.file_to_functions.append((file_path_str, full_name))
                
                # If inside a class, record containment
                if current_class:
                    self.class_contains.append((current_class, func_name, file_path_str))
                
                # Extract function calls inside this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Get called function name
                        if isinstance(child.func, ast.Name):
                            callee = child.func.id
                            self.calls.append((func_name, callee, file_path_str))
                        elif isinstance(child.func, ast.Attribute):
                            # Method call like obj.method()
                            callee = child.func.attr
                            self.calls.append((func_name, callee, file_path_str))
            
            # Extract classes
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                self.classes[class_name] = file_path_str
                
                # Track file contains class
                self.file_to_classes.append((file_path_str, class_name))
                
                current_class = class_name
                
                # Parse class contents (functions inside)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_name = child.name
                        self.functions[func_name] = file_path_str
                        self.class_contains.append((class_name, func_name, file_path_str))
                        self.file_to_functions.append((file_path_str, func_name))
                
                current_class = None
            
            # NEW: Extract imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    self.imports.append((file_path_str, module_name, node.lineno))
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module if node.module else ""
                for alias in node.names:
                    full_module = f"{module_name}.{alias.name}" if module_name else alias.name
                    self.imports.append((file_path_str, full_module, node.lineno))
    
    def parse_repo(self, repo_path: str):
        """Parse all Python files in repository"""
        repo_dir = Path(repo_path)
        
        # Print debug info
        print(f"🔍 Looking for Python files in: {repo_dir.absolute()}")
        
        # Find all .py files
        py_files = list(repo_dir.rglob("*.py"))
        
        # Remove __pycache__ directories
        py_files = [f for f in py_files if '__pycache__' not in str(f)]
        
        print(f"📁 Found {len(py_files)} Python files")
        
        # Print file names for debugging
        for f in py_files:
            print(f"  - {f.name} (in {f.parent.name})")
        
        for py_file in py_files:
            print(f"  🔍 Parsing {py_file.name}")
            self.parse_file(py_file)
        
        print(f"\n✅ Extraction complete:")
        print(f"   - {len(self.files)} files")
        print(f"   - {len(self.functions)} functions")
        print(f"   - {len(self.classes)} classes")
        print(f"   - {len(self.calls)} function calls")
        print(f"   - {len(self.class_contains)} class-function relationships")
        print(f"   - {len(self.imports)} imports")
        
        return {
            'functions': self.functions,
            'classes': self.classes,
            'calls': self.calls,
            'class_contains': self.class_contains,
            'files': self.files,
            'imports': self.imports,
            'file_to_functions': self.file_to_functions,
            'file_to_classes': self.file_to_classes
        }