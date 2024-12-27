import json
import os
from os.path import exists, dirname, join
import subprocess
import sys

import numpy
import pybind11_stubgen

def main():
    import sys
    import os
    import subprocess
    from os.path import pardir, sep
    sys.path.append(os.getcwd())

    # make dir pyclipr
    os.makedirs('pyclipr', exist_ok=True)
    # copy pyclipr.cpython-312-darwin.so into pyclipr
    subprocess.run(['cp', 'pyclipr.cpython-312-darwin.so', 'pyclipr'])
    # copy __init__.py into pyclipr folder
    subprocess.run(['cp', '__init__.py', 'pyclipr'])

    # run the process
    #subprocess.run(['pybind11-stubgen','--output-dir',  os.getcwd(), 'pyclipr'])
    sys.argv = [
        "<dummy>",
        "--output-dir",
        os.getcwd(),
        "pyclipr",
    ]

    pybind11_stubgen.main()


if __name__ == "__main__":
    main()