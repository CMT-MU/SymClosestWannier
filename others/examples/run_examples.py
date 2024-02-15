"""
run examples.
"""
import os
import subprocess


# ================================================== main
def main():
    models = sorted([f for f in os.listdir(".") if os.path.isdir(os.path.join(".", f))])

    for seedname in models:
        print(seedname)
        os.chdir(seedname)
        subprocess.run(f"pw2cw {seedname}", shell=True)
        os.chdir("..")


if __name__ == "__main__":
    main()
