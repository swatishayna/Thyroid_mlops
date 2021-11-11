import os

dirs = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src"
]
for dir in dirs:
    os.mkdir(dir)
    with open(os.path.join(dir, ".gitkeep"), "w") as f:
        pass

files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("src", "__init__.py")
]
for file in files:
    with open(file, "w") as f:
        pass
