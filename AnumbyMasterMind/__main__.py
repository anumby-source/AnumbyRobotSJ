import os
import sys
import argparse

def main(argv):
    print("MasterMind")
    print("Arguments=", len(sys.argv), sys.argv)

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-formes", type=int, default=8, help="number of formes for training")
    argParser.add_argument("-c", "--cell_size", type=int, default=40, help="cell size for figures")
    argParser.add_argument("-d", "--data_size", type=int, default=100, help="data size for traning")

    group = argParser.add_mutually_exclusive_group()
    group.add_argument("--build_figures", action="store_true", help="build figures")
    group.add_argument("-run", action="store_true", help="run")

    argParser.add_argument("-f", "--figure", type=int, default=None, help="figure to build")


    argParser.add_argument("-data", "--load_data", action="store_false", help="load data")
    argParser.add_argument("-model", "--load_model", action="store_false", help="load model")

    args = argParser.parse_args()

    print("cell_size=", args.cell_size)
    print("data_size=", args.data_size)
    print("formes=", args.formes)
    print("build_figures", args.build_figures)
    print("figure", args.figure)
    print("load_data", args.load_data)
    print("load_model", args.load_model)
    print("run", args.run)

    if args.build_figures:
        if args.figure is None:
            print("Formes> Rebuild all figures")
        else:
            print("Formes> Rebuild figure # {}".format(args.figure))
    elif args.run:
        print("Formes> Run with load_data={} load_model={}".format(args.load_data, args.load_model))
    else:
        print("Formes> no action")

if __name__ == "__main__":
    HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
    TOP = os.path.dirname(HERE) + "/AnumbyFormes"
    DATA = TOP + "/data/"

    print("HERE=", HERE, "TOP=", TOP, "DATA=", DATA)
    # main(sys.argv[1:])
