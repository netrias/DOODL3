import time
import keras
import os, h5py
import subprocess
import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.decomposition import PCA
import gpflow
import itertools
import CMGDB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern


class CustomSaver(keras.callbacks.Callback):
    def __init__(self, outdir, epochs=None):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.out_dir = outdir
        if isinstance(epochs, list):
            self.epochs = [1] + epochs
        else:
            self.epochs = list(range(epochs))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs:  # or save after some epoch, each k-th epoch etc.
            # print("Saving", self.model.name, 'epoch', epoch, os.path.join(self.out_dir,
            #                                                               self.model.name + '-{}-{:1.2f}.hdf5'.format(epoch,
            #                                                                                                           logs['val_loss'])))
            self.model.save(os.path.join(self.out_dir,
                                         self.model.name + '-{}-{:1.2f}.hdf5'.format(epoch, logs['val_loss'])))


def train_model_iteratively(baseline_model, X_train, Y_train, X_test, Y_test,
                            outdir, epochs=12, epochs_to_save=None, batch_size=128, num_models=3):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    Y_preds = []
    print("X_train: {}, Y_train: {}".format(X_train.shape, Y_train.shape))
    print("X_test: {}, Y_test: {}".format(X_test.shape, Y_test.shape))
    pred_col_names = ['p' + str(i) for i in range(Y_train.shape[1])]
    for i in range(num_models):
        name = 'model.' + str(i)
        model = baseline_model(name)
        if epochs_to_save is None:
            epochs_to_save = [epochs - 1]
        saver = CustomSaver(outdir, epochs_to_save)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                  validation_data=(X_test, Y_test), callbacks=[saver])
        Y_pred = pd.DataFrame(model.predict(X_test), columns=[pred_col_names])
        Y_pred['class'] = Y_pred.idxmax(axis=1)
        Y_pred['model'] = name
        Y_preds.append(Y_pred)
    # print('appending dataframe')
    Y_predsDF = pd.concat(Y_preds)
    Y_predsDF.to_csv(os.path.join(outdir, 'predictions.csv'))


def get_kernel_names(f):
    datalist = []

    def returnname(name):
        if 'kernel' in name and name not in datalist and 'model_weights' in name:
            return name
        else:
            return None

    looper = 1
    while looper == 1:
        name = f.visit(returnname)
        if name == None:
            looper = 0
            continue
        datalist.append(name)
    return datalist


def get_model_weights(path_to_model_files='/Users/meslami/Documents/GitRepos/deep_chaos/data/logs/cp_larger_random_weights/'):
    model_weights = {}
    for fname in os.listdir(path_to_model_files):
        if 'hdf5' in fname:
            # print(fname)
            f = h5py.File(os.path.join(path_to_model_files, fname), 'r')
            model_weights[fname] = {}
            #         model_weights[fname]['model_id']=fname[:-5]
            model_weights[fname]['model_id'] = fname.split('-')[0]
            model_weights[fname]['epoch'] = fname.split('-')[1]
            model_weights[fname]['val_loss'] = fname.split('-')[2].replace('.hdf5', '')
            # bias_name = f.visit(get_bias)
            # model_weights[fname]['bias']=f[bias_name][:]
            kernel_names = get_kernel_names(f)
            idx_conv, idx_dense = 0, 0
            for kernel_name in kernel_names:
                if 'conv' in kernel_name:
                    model_weights[fname]['conv_kernel_' + str(idx_conv)] = f[kernel_name][:]
                    idx_conv = idx_conv + 1
                if 'dense' in kernel_name:
                    model_weights[fname]['dense_kernel_' + str(idx_dense)] = f[kernel_name][:]
                    idx_dense = idx_dense + 1
    return model_weights


def convert_weight_dict_to_dataframe(model_weights, names_of_interest=['kernel']):
    '''

    :param model_weights: dict, model weights extracted from get_model_weights method
    :param names_of_interest: a list of names of interest. this can be a specific layer(s). if you want all kernel parameters, then just set to 'kernel
    otherwise, supply a list of layers ex. conv_0_kernel for all kernels of layer of first conv layer, or conv for kernels across all conv layers.
    :return: dataframe of file as index, model_id, epoch, val_loss and all parameters indexed
    '''
    protected_cols = ['model_id', 'epoch', 'val_loss']
    df_weights = pd.DataFrame.from_dict(model_weights).T

    def flatten_array(kk):
        return kk.flatten()

    def explode_to_cols(df, col_name):
        if len(df[col_name].iloc[0].shape) > 1:
            df[col_name] = df[col_name].apply(flatten_array)
        return pd.DataFrame(df[col_name].tolist(), index=df.index,
                            columns=[col_name + '_' + str(i) for i in range(len(df[col_name].iloc[0]))])

    df_kernels = [df_weights]
    for name_of_interest in names_of_interest:
        for col_name in df_weights:
            if name_of_interest in col_name:
                df_kernel = explode_to_cols(df_weights, col_name)
                df_kernels.append(df_kernel)

    df_tot = pd.concat(df_kernels, axis=1, join='inner')
    names_to_drop = [col for col in df_weights.columns if col not in protected_cols]
    df_tot.drop(names_to_drop, axis=1, inplace=True)
    return df_tot


def get_weights_with_max_change(df_tot, pattern='kernel'):
    if 'model_id' in df_tot.columns:
        df_tot.set_index('model_id', inplace=True)
    cols = [col for col in df_tot.columns if pattern in col]
    min_epoch = df_tot['epoch'].min()
    max_epoch = df_tot['epoch'].max()

    pca = PCA(n_components=2)
    df_tot[['pca_comp1', 'pca_comp2']] = pca.fit_transform(df_tot[cols])

    df_start = df_tot[df_tot['epoch'] == min_epoch][cols]
    df_end = df_tot[df_tot['epoch'] == max_epoch][cols]

    print(df_start.columns)
    print(len(df_start))
    print(df_end.columns)
    print(len(df_end))

    df_diff = (df_end - df_start) / (float(max_epoch) - float(min_epoch))
    df_diff.fillna(0, inplace=True)

    df_diff['col_with_max_change'] = df_diff[cols].apply(abs).idxmax(axis=1)
    df_diff['col_with_min_change'] = df_diff[cols].apply(abs).idxmin(axis=1)
    df_diff['col_with_max_pos_change'] = df_diff[cols].idxmax(axis=1)
    df_diff['col_with_max_neg_change'] = df_diff[cols].idxmin(axis=1)

    return df_tot, df_diff


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
