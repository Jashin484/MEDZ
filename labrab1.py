import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def f(x, A=1.25313):
    return(0.5+(((np.cos(np.sin(x**2-A**2)))**2-0.5)/(1+0.001*(x**2+A**2))))

def makepath(path):
    if path.is_dir() == False:
        Path.mkdir("results")

def writefile(x, y, count):
    datax=[]
    datay=[]
    for n in range(count):
        datax+=[x[n]]
        datay+=[y[n]]
    result={"x": datax, "y": datay}
    with open(Path("results","results1.json"), mode="w") as file:
        file.write(json.dumps(result, indent=4))

def draw(x, y):
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    count=200
    xmin=-100
    xmax=100
    xdata=np.linspace(xmin, xmax, count)
    ydata=f(xdata)
    makepath(Path("results"))
    writefile(xdata, ydata, count)
    draw(xdata, ydata)