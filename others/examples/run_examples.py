"""
run examples.

the examples without seedname.amn (removed to reduce the repository size) are skipped.
"""

import os
import subprocess


# ================================================== main
def main():
    models = sorted([f for f in os.listdir(".") if os.path.isdir(os.path.join(".", f))])

    for seedname in models:
        if not os.path.isfile(os.path.join(seedname, f"{seedname}.amn")):
            print(f"{seedname}: skipped ({seedname}.amn is not found)")
            continue
        print(seedname)
        os.chdir(seedname)
        subprocess.run(f"pw2cw {seedname}", shell=True)
        os.chdir("..")


if __name__ == "__main__":
    main()
