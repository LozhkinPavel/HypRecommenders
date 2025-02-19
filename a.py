import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--emb_dims", type=int, nargs='+', default=[32, 64])

args = parser.parse_args()

print(vars(args))