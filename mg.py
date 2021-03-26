
import time
import numpy as np
import gpflow
import itertools
import CMGDB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import subprocess
from IPython.display import Image


class gp_tensorflow:
    __model = None

    def __init__(self, X, Y):
        if gp_tensorflow.__model is None:
            # Choose kernel
            kernel = gpflow.kernels.Matern52()
            # Define model
            gp_tensorflow.__model = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=None)
            # Choose optimizer
            optimizer = gpflow.optimizers.Scipy()

            optimizer.minimize(gp_tensorflow.__model.training_loss, gp_tensorflow.__model.trainable_variables,
                               options=dict(maxiter=100))

    def get_model(self):
        if gp_tensorflow.__model is None:
            raise ValueError('Model does not exist. Instantiate first.')
        return gp_tensorflow.__model


class gp:
    __model = None

    def __init__(self, X, Y):
        if gp.__model is None:
            # fit Gaussian Process with dataset X_train, Y_train
            kernel = RBF(0.5, (0.01, 2)) + WhiteKernel()
            gp.__model = GaussianProcessRegressor(kernel=kernel)
            gp.__model.fit(X, Y)

    def get_model(self):
        if gp.__model is None:
            raise ValueError('Model does not exist. Instantiate first.')
        return gp.__model


# Returns corner points of a rectangle
def _CornerPoints(rect):
    dim = int(len(rect) / 2)
    # Get list of intervals
    list_intvals = [[rect[d], rect[d + dim]] for d in range(dim)]
    # Get points in the cartesian product of intervals
    X = [list(u) for u in itertools.product(*list_intvals)]
    return X


# Returns center point of a rectangle
def _CenterPoint(rect):
    dim = int(len(rect) / 2)
    x_center = [(rect[d] + rect[dim + d]) / 2 for d in range(dim)]
    return [x_center]


# Return sample points in rectangle
def _SamplePoints(lower_bounds, upper_bounds, num_pts):
    # Sample num_pts in dimension dim, where each
    # component of the sampled points are in the
    # ranges given by lower_bounds and upper_bounds
    dim = len(lower_bounds)
    X = np.random.uniform(lower_bounds, upper_bounds, size=(num_pts, dim))
    return list(X)


# Map that takes a rectangle and returns a rectangle
def _BoxMap(f, rect, mode='corners', padding=False, num_pts=10):
    dim = int(len(rect) / 2)
    if mode == 'corners':  # Compute at corner points
        X = _CornerPoints(rect)
    elif mode == 'center':  # Compute at center point
        padding = True  # Must be true for this case
        X = _CenterPoint(rect)
    elif mode == 'random':  # Compute at random point
        # Get lower and upper bounds
        lower_bounds = rect[:dim]
        upper_bounds = rect[dim:]
        X = _SamplePoints(lower_bounds, upper_bounds, num_pts)
    else:  # Unknown mode
        return []
    # Evaluate f at point in X
    Y = [f(x) for x in X]
    # Get lower and upper bounds of Y
    Y_l_bounds = [min([y[d] for y in Y]) - ((rect[d + dim] - rect[d]) if padding else 0) for d in range(dim)]
    Y_u_bounds = [max([y[d] for y in Y]) + ((rect[d + dim] - rect[d]) if padding else 0) for d in range(dim)]
    f_rect = Y_l_bounds + Y_u_bounds
    return f_rect


def compute_morse_graph(data, phase_subdiv=5):
    '''
    Method to compute the Morse Graph using an sklearn GP (see class gp)
    :param data: dataframe of parameters that you want to compute Morse Graph on.
    :param phase_subdiv:
    :return:
    '''

    def _f(X):
        return gp_instance.get_model().predict(np.asarray([X]))[0]  # Need to change to predict_f for tensorflow

    # Define box map for f
    def _F(rect):
        return _BoxMap(_f, rect, padding=True)

    cols = [x for x in data.columns if 'kernel' in x]
    # Define the parameters for CMGDB
    lower_bounds = [data[x].min() - 0.5 for x in data.columns if 'kernel' in x]
    upper_bounds = [data[x].max() + 0.5 for x in data.columns if 'kernel' in x]

    ##INSTANTIATE!!!! gp here!
    X1 = data[data['epoch'] == data['epoch'].min()][cols]
    Y1 = data[data['epoch'] == data['epoch'].max()][cols]
    gp_instance = gp(X1, Y1)
    # print(type(X1.to_numpy()))
    print(gp_instance.get_model().predict(X1.to_numpy())[0])  # Need to change to predict_f for tensorflow
    morse_fname = 'morse_sets.csv'

    model = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, _F)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    return morse_graph, map_graph


def compute_morse_graph_with_gpflow_gp(data, phase_subdiv=5):
    '''
    Method to compute the Morse Graph using a gpflow GP (see class gp_tensorflow)
    :param data: dataframe of parameters that you want to compute Morse Graph on.
    :param phase_subdiv:
    :return:
    '''
    start = time.time()

    def _f(X):
        Y, var = gp_instance.get_model().predict_f(np.array([X]))
        return np.array(Y)[0]

    # Define box map for f
    def _F(rect):
        return _BoxMap(_f, rect, padding=True)

    cols = [x for x in data.columns if 'kernel' in x]
    # Define the parameters for CMGDB
    lower_bounds = [data[x].min() - 0.5 for x in data.columns if 'kernel' in x]
    upper_bounds = [data[x].max() + 0.5 for x in data.columns if 'kernel' in x]

    ##INSTANTIATE!!!! gp here!
    X1 = data[data['epoch'] == data['epoch'].min()][cols]
    Y1 = data[data['epoch'] == data['epoch'].max()][cols]
    gp_instance = gp_tensorflow(X1, Y1)
    # print(type(X1.to_numpy()))
    # print(gp_instance.get_model().predict_f(X1.to_numpy())[0])  # Need to change to predict_f for tensorflow
    morse_fname = 'morse_sets.csv'

    model = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, _F)
    morse_graph, map_graph = CMGDB.ComputeMorseGraph(model)
    print("Duration of compute_morse_graph_with_gpflow_gp: {} minutes".format((time.time() - start) / 60))
    return morse_graph, map_graph


def compute_order_retraction(morse_graph, map_graph, title):
    # Save MVMap to file for CMGDB_Retract
    with open('CMGDB_mvmap.txt', 'w') as outfile:
        outfile.write('%d\n' % map_graph.num_vertices())
        for k in range(0, map_graph.num_vertices()):
            l = len(map_graph.adjacencies(k))
            outfile.write('%d ' % k)
            outfile.write('%d ' % l)
            for item in map_graph.adjacencies(k):
                outfile.write('%d ' % item)
            outfile.write('\n')

    subprocess.call(['./CMGDB_RETRACT'])

    # Load retraction from file
    with open('CMGDB_retract.txt', 'r') as infile:
        retract_indices = []
        retract_tiles = []
        for i in range(0, map_graph.num_vertices()):
            index, tile = [int(x) for x in next(infile).split()]
            retract_indices.append(index)
            retract_tiles.append(tile)

    # # Display Hasse diagram using Graphviz to compare to Morse graph
    # # The index labels of the Morse sets may have changed!!
    # subprocess.call("dot -Tpng Hasse.dot -o Hasse.png")
    # Image("Hasse.png")

    # Plot order retraction: you may need to change the colors to match CMGDB output
    bx = morse_graph.phase_space_box(0)
    bx_width = bx[2] - bx[0]
    bx_height = bx[3] - bx[1]
    # print(bx_width);
    # print(bx_height);
    fig, ax = plt.subplots(figsize=(7 ,7))
    # ax.set(xlim=(lower_bounds[0],upper_bounds[0]),ylim=(lower_bounds[1],upper_bounds[1]));
    ax.set(xlim=[-2, 2], ylim=[-2, 2])
    boxes_array = []
    for j in range(0, morse_graph.num_vertices()):
        boxes_array.append([])
    for i in range(0, map_graph.num_vertices()):
        for j in range(0, morse_graph.num_vertices()):
            if retract_tiles[i] == j:
                bx = morse_graph.phase_space_box(retract_indices[i])
                rect = Rectangle(((bx[2] + bx[0]) / 2, (bx[3] + bx[1]) / 2), bx_width, bx_height);
                boxes_array[j].append(rect)
    for j in range(0, morse_graph.num_vertices()):
        # print(len(boxes_array[j]))
        if j == 0:
            # color='b'
            color = 'c'  # I changed the color because the alpha wan't working for some strange reason
        if j == 1:
            color = 'm'
        if j == 2:
            color = 'k'
        if j == 3:
            color = 'g'
        if j == 4:
            color = 'k'
        if j == 5:
            color = 'k'
        if j == 6:
            color = 'k'
        if j == 7:
            color = 'k'
        if j == 8:
            color = 'k'
        pc = PatchCollection(boxes_array[j], facecolor=color, alpha=0.3, edgecolor='none')
        ax.add_collection(pc)

    for j in range(0, morse_graph.num_vertices()):
        # print(len(morse_graph.morse_set(j)))
        boxes = []
        for index in morse_graph.morse_set(j):
            bx = morse_graph.phase_space_box(index)
            rect = Rectangle(((bx[2] + bx[0]) / 2, (bx[3] + bx[1]) / 2), bx_width, bx_height);
            boxes.append(rect)
        if j == 0:
            color = 'b'
        if j == 1:
            color = 'r'
        if j == 2:
            color = 'g'
        if j == 3:
            color = 'g'
        if j == 4:
            color = 'k'
        if j == 5:
            color = 'k'
        if j == 6:
            color = 'k'
        if j == 7:
            color = 'k'
        if j == 8:
            color = 'k'
        pc2 = PatchCollection(boxes, facecolor=color, alpha=1.0, edgecolor='none')
        ax.add_collection(pc2)

    plt.title(title)
    plt.show()

def compute_mse(Y_pred,Y_actuals):
    return ((Y_pred - Y_actuals) ** 2).mean()

def plot_actuals_vs_predicted(X1,Y_pred,Y1,num_components=2):
    '''
    Plot actuals vs predicted
    :param X1: Training dataframe
    :param Y_pred: Predicted data in numpy format.
    Usually: gp_instance.get_model().predict(X1.to_numpy()) OR gpt.get_model().predict_f(X1.to_numpy())[0].numpy()
    :param Y1: Actual dataframe
    :param num_components: number of components
    :return:
    '''
    for i in range(num_components):
        plt.plot(X1.to_numpy()[:, i], Y_pred[:, i], 'yo',
                 label='Predicted-Dim0')
        plt.plot(X1.to_numpy()[:, i], Y1.to_numpy()[:, i], 'b.', label='Actual-Dim1)');

    plt.legend(loc="upper left");


