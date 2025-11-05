import argparse
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

def loadfile(input):
    with open(Path("results", f"{input}"), mode="r") as file:
        data = json.load(file)
    return data["x"], data["y"]

def draw(x, y, ystep = None):
    plt.plot(x, y)
    plt.grid(True)
    if ystep != None:
        plt.yticks(np.arange(min(y), max(y)+ystep, step=ystep))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type = str)
    parser.add_argument("--ystep", "-ys", "-y_s", type = float)
    arguments = parser.parse_args()

    xdata, ydata = loadfile(arguments.inputfile)

    draw(xdata, ydata, arguments.ystep)