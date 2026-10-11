import sys
import os

print(f"Executing Python: {sys.executable}")
print(f"Python Version: {sys.version.split()[0]}")

packages = ["torch", "tensorflow", "keras", "transformers", "pandas", "numpy", "sklearn", "fastapi", "uvicorn", "pydantic", "pytest"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  {pkg:<15}: {version}")
    except ImportError as e:
        print(f"  {pkg:<15}: NOT INSTALLED")
