import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix")
args = parser.parse_args()
print(args.prefix)