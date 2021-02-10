import keras
import os, h5py
import pandas as pd
import numpy as np

class CustomSaver(keras.callbacks.Callback):
    def __init__(self,outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('set outdir',outdir)
        self.out_dir = outdir

    def on_epoch_end(self, epoch,logs=None):
        if epoch == 1 or epoch==11:  # or save after some epoch, each k-th epoch etc.
            print("Saving",self.model.name,'epoch',epoch)
            print('Saving')
            print(os.path.join(self.out_dir,
                            self.model.name + '-{}-{:1.2f}.hdf5'.format(epoch,logs['val_loss'])))
            self.model.save(os.path.join(self.out_dir,
                            self.model.name + '-{}-{:1.2f}.hdf5'.format(epoch,logs['val_loss'])))


def train_model_iteratively(baseline_model,X_train,Y_train,X_test,Y_test,outdir,batch_size=128):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    Y_preds = []
    print("Train:",X_train.shape,Y_train.shape)
    print("Test:",X_test.shape,Y_test.shape)
    pred_col_names = ['p'+str(i) for i in range(Y_train.shape[1])]
    for i in range(20):
        name = 'model.' + str(i)
        print(name)
        model = baseline_model(name)
        # saver = CustomSaver(outdir)
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=12, verbose=0,
                                     validation_data=(X_test, Y_test) )#,callbacks=[saver]
        Y_pred = pd.DataFrame(model.predict(X_test), columns=[pred_col_names])
        Y_pred['class'] = Y_pred.idxmax(axis=1)
        Y_pred['model'] = name
        Y_preds.append(Y_pred)
    print('appending dataframe')
    Y_predsDF = pd.concat(Y_preds)
    Y_predsDF.to_csv(os.path.join(outdir,'predictions.csv'))

def get_bias(name):
    if 'bias' in name:
        print(name)
        return name



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
    model_weights={}
    for fname in os.listdir(path_to_model_files):
        if 'hdf5' in fname:
            print(fname)
            f = h5py.File(os.path.join(path_to_model_files,fname), 'r')
            model_weights[fname]={}
    #         model_weights[fname]['model_id']=fname[:-5]
            model_weights[fname]['model_id']=fname.split('-')[0]
            model_weights[fname]['epoch']=fname.split('-')[1]
            model_weights[fname]['val_loss']=fname.split('-')[2].replace('.hdf5','')
            # bias_name = f.visit(get_bias)
            # model_weights[fname]['bias']=f[bias_name][:]
            kernel_names = get_kernel_names(f)
            idx_conv,idx_dense = 0,0
            for kernel_name in kernel_names:
                if 'conv' in kernel_name:
                    model_weights[fname]['conv_kernel_'+str(idx_conv)]=f[kernel_name][:]
                    idx_conv = idx_conv+1
                if 'dense' in kernel_name:
                    model_weights[fname]['dense_kernel_'+str(idx_dense)]=f[kernel_name][:]
                    idx_dense = idx_dense+1
    return model_weights

def convert_weight_dict_to_dataframe(model_weights):
    protected_cols = ['model_id','epoch','val_loss']
    df_weights = pd.DataFrame.from_dict(model_weights).T

    def flatten_array(kk):
        return kk.flatten()

    def explode_to_cols(df, col_name):
        if len(df[col_name].iloc[0].shape) > 1:
            df[col_name] = df[col_name].apply(flatten_array)
        return pd.DataFrame(df[col_name].tolist(), index=df.index,
                            columns=[col_name + '_' + str(i) for i in range(len(df[col_name].iloc[0]))])
    df_kernels = [df_weights]
    for col_name in  df_weights:
        if 'kernel' in col_name:
            df_kernel = explode_to_cols(df_weights, col_name)
            df_kernels.append(df_kernel)
    df_tot = pd.concat(df_kernels, axis=1, join='inner')
    names_to_drop = [col for col in df_weights.columns if col not in protected_cols]
    df_tot.drop(names_to_drop,axis=1,inplace=True)
    return df_tot