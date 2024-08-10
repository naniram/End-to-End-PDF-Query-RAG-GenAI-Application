# to create Project structure

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

list_of_files = [
    f"src/__init__.py",
    f"src/helper.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]


for filepath in list_of_files:
    filepath = Path(filepath)   # adjusts path / for linux \ for windows automatically
    filedir, filename = os.path.split(filepath) #extracting file directory and file from the filepath

    if filedir != "":   # if not empty make these directories
        os.makedirs(filedir, exist_ok=True) #if already exists, ignore
        logging.info(f"creating_directory; {filedir} for the file: {filename}")

    # To create file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")  #if the file already exists

# now run this python file from the terminal