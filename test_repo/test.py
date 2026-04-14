import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CodeParser:
    def __init__(self):
        self.functions = {}  # {function_name: file_path}
        self.classes = {}  # {class_name: file_path}
        self.calls = []  # [(caller, callee, file_path)]
        self.class_contains = []  # [(class_name, function_name, file_path)]

    def parse_file(self, filepath: Path):
        """Parse a single Python file and extract structure"""
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            print(f"  ⚠️  Error parsing {filepath}: {e}")
            return

        current_class = None

        for node in ast.walk(tree):
            # Extract functions
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                full_name = f"{func_name}"
                self.functions[full_name] = str(filepath)

                # If inside a class, record containment
                if current_class:
                    self.class_contains.append(
                        (current_class, func_name, str(filepath))
                    )

                # Extract function calls inside this function
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        # Get called function name
                        if isinstance(child.func, ast.Name):
                            callee = child.func.id
                            self.calls.append((func_name, callee, str(filepath)))
                        elif isinstance(child.func, ast.Attribute):
                            # Method call like obj.method()
                            callee = child.func.attr
                            self.calls.append((func_name, callee, str(filepath)))

            # Extract classes
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                self.classes[class_name] = str(filepath)
                current_class = class_name

                # Parse class contents (functions inside)
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        func_name = child.name
                        self.functions[func_name] = str(filepath)
                        self.class_contains.append(
                            (class_name, func_name, str(filepath))
                        )

                current_class = None

    def parse_repo(self, repo_path: str):
        """Parse all Python files in repository"""
        repo_dir = Path(repo_path)

        # Print debug info
        print(f"🔍 Looking for Python files in: {repo_dir.absolute()}")

        # Find all .py files (removed the test filter for now)
        py_files = list(repo_dir.rglob("*.py"))

        # Remove __pycache__ directories
        py_files = [f for f in py_files if "__pycache__" not in str(f)]

        print(f"📁 Found {len(py_files)} Python files")

        # Print file names for debugging
        for f in py_files:
            print(f"  - {f.name} (in {f.parent.name})")

        for py_file in py_files:
            print(f"  🔍 Parsing {py_file.name}")
            self.parse_file(py_file)

        print(f"\n✅ Extraction complete:")
        print(f"   - {len(self.functions)} functions")
        print(f"   - {len(self.classes)} classes")
        print(f"   - {len(self.calls)} function calls")
        print(f"   - {len(self.class_contains)} class-function relationships")

        return {
            "functions": self.functions,
            "classes": self.classes,
            "calls": self.calls,
            "class_contains": self.class_contains,
        }
