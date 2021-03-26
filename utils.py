import keras
import os, h5py
import pandas as pd
from sklearn.decomposition import PCA

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
