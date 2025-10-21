import os
import glob
import importlib

# get directory
dir = os.path.dirname(__file__)

# get all .py files
files = glob.glob(os.path.join(dir, "*.py"))
modules = [
    os.path.splitext(os.path.basename(f))[0]
    for f in files
    if not f.endswith("__init__.py")
]

# import each module
for module in modules:
    importlib.import_module(f".{module}", package=__name__)

__all__ = modules

# success message (for testing)
# print("rice2025 package imported successfully")