import numpy as np
import matplotlib.pyplot as plt


def function(x, D = 2):
    y2 = 0
    for i in range(D):
        y2 += (x[i])**2 - (10 * np.cos(2*np.pi*x[i]))
    y = (10 * D) + y2
    return y

def graph_2d(xmin, xmax, x10 = None, x20 = None, count = 200):
    xgrid = [0, 0]

    if (x10 == None) & (x20 != None):
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x
        xgrid[1] = x20

        y = function(xgrid)
        return xgrid[0], y
    
    elif (x20 == None) & (x10 != None):
        x = np.linspace(xmin, xmax, count)

        xgrid[0] = x10
        xgrid[1] = x

        y = function(xgrid)
        return xgrid[1], y
    
    else:
        return None, None

def graph_3d(x1min, x1max, x2min, x2max, count = 200):
    xgrid = [0, 0]

    x1 = np.linspace(x1min, x1max, count)
    x2 = np.linspace(x2min, x2max, count)

    xgrid[0], xgrid[1] = np.meshgrid(x1, x2)

    y = function(xgrid)
    return xgrid[0], xgrid[1], y

def draw(x1min, x1max, x2min, x2max):
    fig = plt.figure(figsize=[16, 9])

    yn = function([x['10'], x['20']])

    x1, x2, y = graph_3d(x1min, x1max, x2min, x2max)

    axes_1 = fig.add_subplot(2, 2, 1, projection='3d')
    axes_1.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)")
    axes_1.plot_surface(x1, x2, y, cmap="twilight")
    axes_1.scatter(x['10'], x['20'], yn, color='black', marker='o')
    axes_1.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)")

    axes_2 = fig.add_subplot(2, 2, 2, projection='3d')
    axes_2.set(xlabel="x\u2081", ylabel="x\u2082", zlabel="y = f(x\u2081, x\u2082)")
    axes_2.view_init(elev=90, azim=0)
    axes_2.plot_surface(x1, x2, y, cmap="twilight")
    axes_2.scatter(x['10'], x['20'], yn, color='black', marker='o')
    axes_2.text(x['10'], x['20'], yn, "y = f(x\u2081\u2080, x\u2082\u2080)")
    
    x1, y = graph_2d(x1min, x1max, x20 = x['20'])

    axes_3 = fig.add_subplot(2, 2, 3)
    axes_3.set(xlabel = ("x\u2081"), ylabel = ("y = f(x\u2081)"))
    axes_3.plot(x1, y, color = "#735184")
    axes_3.scatter(x['10'], yn, color='black', marker='o')
    axes_3.text(x['10'], yn, "y = f(x\u2081\u2080)")

    x2, y = graph_2d(x2min, x2max, x10 = x['10'])

    axes_4 = fig.add_subplot(2, 2, 4)
    axes_4.set(xlabel = ("x\u2082"), ylabel = ("y = f(x\u2082)"))
    axes_4.plot(x2, y, color = "#735184")
    axes_4.scatter(x['20'], yn, color='black', marker='o')
    axes_4.text(x['20'], yn, "y = f(x\u2082\u2080)")

    plt.show()

if __name__ == "__main__":
    x1min = -5.12
    x1max = 5.12
    x2min = -5.12
    x2max = 5.12
    x = {'10': 0. , '20': 0.}

    draw(x1min, x1max, x2min, x2max)