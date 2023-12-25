import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=float, nargs='+')
args = parser.parse_args()
import numpy as np
print(np.mean(args.f) * 100)
print(np.std(args.f) * 100)
