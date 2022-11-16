from genericpath import exists
import glob
from importlib.resources import path
import pandas as pd
import numpy as np
from LightCurve import CRTS_VS_LightCurve
import random
import re
from astropy import units
from astropy.time import Time
import pickle
import matplotlib.pyplot as plt
import itertools
import os

numlabels = {1:'RRab', 2:'RRc', 3:'RRd', 4:'Blazhko', 5:'Ecl', 6:'EA', 7:'ROT', 
            8:'LPV', 9:'DS', 10:'ACEP', 11:'Miscellaneous', 12:'Cep-2', 13:'LMC-Cep'}

def plot_confusion_matrix(cm, classes_types,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    fig = plt.figure(figsize=(12,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cb=plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes_types))
    plt.xticks(tick_marks, classes_types, rotation=45)
    plt.yticks(tick_marks, classes_types)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if (cm[i, j] < 0.01) or (cm[i,j] >= 0.75)  else "black",fontsize=18)

    plt.ylabel('True label',fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)
    plt.tight_layout()
    return fig

def cm_plot(conf_mat, classes_types, classifier_model,  nClasses,cmap=plt.cm.Reds, plot_title=None):
    
    fig = plot_confusion_matrix(conf_mat, classes_types, normalize=True, title='Confusion matrix for ' + str(classifier_model) )
    # plt.savefig(plot_title +'_CM.pdf',bbox_inches = 'tight',pad_inches = 0.1)
    fig.show()
    return fig
    
def times_to_lags(T):
    """(N x n_step) matrix of times -> (N x n_step) matrix of lags.
    First time is assumed to be zero.
    """
    assert T.ndim == 2, "T must be an (N x n_step) matrix"
    return np.c_[np.zeros(T.shape[0]), np.diff(T, axis=1)]

def preprocess(X, m_max=np.Inf):
    '''
    对time lag只除了标准差
    X[samples : timesteps : dims]
    '''
    X = X.copy()

    X[:, :, 0] = times_to_lags(X[:, :, 0])

    mag_means = np.atleast_2d(np.nanmean(X[:, :, 1], axis=1)).T
    X[:, :, 1] -= mag_means

    mag_scales = np.atleast_2d(np.nanstd(X[:, :, 1], axis=1)).T
    X[:, :, 1] /= mag_scales

    # lag_means = np.atleast_2d(np.nanmean(X[:, :, 0], axis=1)).T
    # X[:, :, 0] -= mag_means

    lag_scales = np.atleast_2d(np.nanstd(X[:, :, 0], axis=1)).T
    X[:, :, 0] /= lag_scales

    X[np.isnan(X)] = 0

    return X


def generate_batch(X_sequence=[], X_image=[], X_feature=[], Y=[], batch_size=32):
    idx = 0
    batch_num = len(Y)/batch_size
    batch_x_sequence = batch_x_image = batch_x_feature = None
    while idx < batch_num:
        batch_x_sequence = X_sequence[idx * batch_size : (idx + 1) * batch_size]
        batch_x_image = X_image[idx * batch_size : (idx + 1) * batch_size]
        batch_x_feature = X_feature[idx * batch_size : (idx + 1) * batch_size]
        batch_y = Y[idx * batch_size : (idx + 1) * batch_size]
        yield batch_x_sequence, batch_x_image, batch_x_feature, batch_y
        idx += 1

def get_meta_information(file_path):
    info = pd.read_csv(file_path, delim_whitespace=True, index_col="File_Name", 
        names=["File_Name", "RA", "Dec", "Period", "V_CSS", "Npts", 
        "V_amp", "Type", "Prior_ID","extra1", "extra2", "extra3"])   
    return info

def dataset_split(test_ratio, val_ratio, seed=0):
    ''' 
    legacy
    没考虑到交叉验证需要，应该先load（预处理）再划分
    '''
    class_folders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    total_train_names = []
    total_val_names = []
    total_test_names = []
    for i, folder in enumerate(class_folders):
        random.seed(seed+i)
        name_list = glob.glob('data/original_data/type/%s/*.dat'%folder)
        num_test = int(len(name_list)*test_ratio)
        num_val = int(len(name_list)*val_ratio)       
        test_names = random.sample(name_list, num_test)
        val_and_train_names = list(set(name_list) - set(test_names))
        random.seed(seed+i+2341)
        val_names = random.sample(val_and_train_names, num_val)
        train_names = list(set(val_and_train_names) - set(val_names))
        total_train_names.append(train_names)
        total_val_names.append(val_names)
        total_test_names.append(test_names)
    print('dataset split finished')
    return total_train_names, total_val_names, total_test_names

def load_data(file_name_list, GP_model=False):
    '''
    读取单个类的数据
    将原始数据读取为LightCurve对象，并进行预处理
    '''
    info_file = 'data/SSS_Per_Tab.txt'
    info = get_meta_information(info_file)
    lc_list = []
    for name in file_name_list:
        split_name = re.match(r'(data/original_data/type/)(\d{1,2})(/)(\d{11,14})(\w{0,15}).dat',name)
        class_label = split_name.group(2)
        lc = CRTS_VS_LightCurve()
        lc.class_label = class_label
        lc.read_CRTS_dat(name, id=split_name.group(4))
        period = info.loc[int(lc.id), 'Period']
        lc.fold(period, normalize_phase=False)
        lc.clean()
        if GP_model == True:
            try:
                lc.fit_GP_model()
            except:
                print('there is an error when fit GP_model, class_label=%s, id=%s'%(lc.class_label,lc.id))
                lc.show()
        lc_list.append(lc)
    print('load %d files'%len(file_name_list))
    return lc_list

def load_original_data_without_split(GP_limit=5000):
    '''
    读取所有数据并进行预处理，并保存为对象
    GP_limit : 预处理时是否拟合GP模型的类中样本数量上限
    和下面几个是一个整体，还未完成
    '''
    class_folders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dataset = []
    for i, folder in enumerate(class_folders):
        name_list = glob.glob('data/original_data/type/%s/*.dat'%folder)
        lc_list = load_data(name_list, GP_model = True if len(name_list) <= GP_limit else False)
        dataset.append(lc_list)
    f = open('data/original_dataset','wb')
    pickle.dump(dataset, f)
    f.close
    return

def dataset_split_after_load():
    '''
    保存划分好的未增强数据集，划分中分类保存
    有需要再写吧，先用legacy的，十倍时间就十倍时间吧
    增加了uncertainty map之后，这些节省的时间好像不重要了，先搁置吧
    '''
    f = open('data/original_dataset','wb')    
    original_dataset = pickle.load(f)
    f.close()
    pass

def dataset_split_K_fold_after_load(K=10):
    '''
    保存划分好的未增强数据集，划分中分类保存
    '''
    pass

def load_original_data(test_ratio, val_ratio, seed=0, GP_model=False):
    '''
    legacy
    没考虑到交叉验证需要，应该先load（预处理）再划分
    好吧其也不用改了，毕竟生成uncertainty map的时间也很多
    '''
    train_names, val_names, test_names = dataset_split(test_ratio, val_ratio, seed)
    print('class labels are [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]')
    print('loading train data')
    train_data = [load_data(train_names[i], GP_model=GP_model) for i in range(0,11)]
    print('loading val data')
    val_data = [load_data(val_names[i], GP_model=GP_model) for i in range(0,11)]
    print('loading test data')
    test_data = [load_data(test_names[i], GP_model=GP_model) for i in range(0,11)]
    original_dataset = (train_data, val_data, test_data)
    f = open('data/original_dataset_%.2f_%.2f_%.2f'%(1-test_ratio-val_ratio, test_ratio, val_ratio), 'wb')
    pickle.dump(original_dataset, f)
    f.close()
    return original_dataset


def get_sample_cadence(flatten_data, period):
    '''
    randomly select a observation cadence from light curves in trainning dataset,
    for generate simulated light curve

    flatten_data: [samples] <- [classes : samples] 
    '''
    idx = random.randint(0, len(flatten_data)-1)
    time_fmt = flatten_data[idx].time_fmt
    time_cadence =  np.array(flatten_data[idx].data[time_fmt])
    time_cadence.sort()
    phase =(time_cadence - time_cadence[0]) % period
    return phase

def flat_data(data):
    '''
    [classes : samples] -> [samples]
    '''
    flatten_data = []
    for class_type in data:
        for sample in class_type:
            flatten_data.append(sample)
    return flatten_data

def GP_augment(light_curves, flatten_data, size):
    '''
    ---
    flatten_data : 用于随机抽取观测cadence，以生成模拟观测
    '''
    count = 0
    batch = 0
    num_samples = len(light_curves)
    initial_data = light_curves.copy()
    error_count = 0
    while count < size-num_samples:
        batch += 1
        print('augmenting')
        for lc in initial_data:
            x = get_sample_cadence(flatten_data, lc.period)
            # x = np.random.uniform(low=0,high=lc.period, size=len(lc))
            # x = np.sort(x)
            try:
                simu_lc = lc.generate_GP_simulation(x, phase_shift_ratio=random.uniform(0,1))
            except:
                print('there is an error when generate_GP_simulation, class_label=%s, id=%s'%(lc.class_label,lc.id))
                lc.show()
                error_count += 1
                continue
            simu_lc.class_label = lc.class_label
            simu_lc.id = lc.id + 'generate%d'%batch
            light_curves.append(lc)
            count += 1
            if count == size-num_samples:
                break
        print('batch=%d, count=%d'%(batch,count))
    return


def get_extra_features():
    '''
    feature的预处理还是留着后面吧
    '''
    namelist = glob.glob('data/features/*/*')
    frames = []
    for name in namelist:
        df = pd.read_csv(name, index_col='File_Name')
        frames.append(df)
    result = pd.concat(frames)
    # for column in ['0', '1', '2', '3', '4', '5']:
    #     mean = np.mean(result[column])
    #     std = np.std(result[column])
    #     result[column] = (result[column]-mean)/std
    return result 

def save_dataset_multi_input(data, name, feature=True, image=True, feature_type='physical'):
    print('saving data')
    X_sequence = []
    Y = []
    flatten_data = flat_data(data)
    random.shuffle(flatten_data)
    for lc in flatten_data:
        X_sequence.append(np.column_stack((np.array(lc.data['phase']), 
                                np.array(lc.data[lc.measurement ]), 
                                np.array(lc.data[lc.error]))))                                
        Y.append(int(lc.class_label))
    data=[X_sequence]
    if feature == True:
        if feature_type != 'physical':
            feature_df = get_extra_features()
        X_feature = []
        for lc in flatten_data:
            if feature_type == 'physical':
                X_feature.append([lc.period, lc.amplitude])
            else:
                X_feature.append(list(feature_df.loc[int(lc.id), ['0', '1', '2', '3', '4', '5']]))
        data.append(X_feature)
    image_count = 0
    if image == True:
        X_image = []
        X_uncertainty_map = []
        for lc in flatten_data:
            X_image.append(np.array(lc.to_image()))
            X_uncertainty_map.append(lc.to_uncertainty_map())
        data.append(X_image)
        data.append(X_uncertainty_map)
        image_count += 1
        print('generated %d images'%image_count)
    data.append(Y)
    f = open(name, 'wb')
    pickle.dump(data, f)
    f.close()
    return

def compute_weight(data):
    weight = []
    for class_type in data:
        weight.append(len(class_type))
    weight = 1. / np.array(weight)
    weight = weight / np.max(weight)
    return weight

def max_class_number(data):
    class_numbers = []
    for class_type in data:
        class_numbers.append(len(class_type))
    return np.max(np.array(class_numbers))

def create_dataset(original_dataset, class_size, prefix='',aug_val=True, down_sample=False, 
            down_sample_size=None, instance=0, save_weight=False, image=False):
    '''
    original_dataset : [classes : samples]
    down_sample_size must be greater than class_size
    '''
    f1 = open(original_dataset, 'rb')
    train_data, val_data, test_data = pickle.load(f1)
    f1.close()
    flatten_train_data = flat_data(train_data)
    split_file_name = re.match(r'(data/original_dataset)(.*)', original_dataset)
    # suffix = split_file_name.group(2) + '_aug_to_%d'%class_size + '_down_sample_%s'%str(down_sample)
    suffix = prefix + split_file_name.group(2) + '_aug_to_%d'%class_size + '_down_sample_%s'%str(down_sample)
    
    if not(os.path.exists('data/split'+suffix+'_instance0-9')): 
        os.mkdir('data/split'+suffix+'_instance0-9')
    if not(os.path.exists('data/split'+suffix+'_instance0-9/val_data')):
        if aug_val == True:
            max_val_class_number = max_class_number(val_data) 
            #这里方便起见没有设置可选的验证集增强数目，如果要加上，必须另外增加验证集下采样，以保证均衡
            for class_type in val_data:
                if len(class_type) < max_val_class_number:
                    GP_augment(class_type, flatten_train_data, max_val_class_number)
        save_dataset_multi_input(val_data, 'data/split'+suffix+'_instance0-9/val_data', image=image)
    for class_type in train_data:
        if len(class_type) < class_size:
            GP_augment(class_type, flatten_train_data, class_size)   
    if down_sample == True:
        if down_sample_size == None:
            down_sample_size = class_size
        for i, class_type in enumerate(train_data):
            if len(class_type) > down_sample_size:
                train_data[i] = random.sample(class_type, down_sample_size)
    save_dataset_multi_input(train_data, 'data/split'+suffix+'_instance0-9/train_data%d'%instance, image=image)

    if not(os.path.exists('data/split'+suffix+'_instance0-9/test_data')):
        save_dataset_multi_input(test_data, 'data/split'+suffix+'_instance0-9/test_data', image=image)

    # if not(os.path.exists('data/split'+suffix+'_instance0-9/aug_test_data')):
    #     max_test_class_number = max_class_number(test_data) 
    #     for class_type in test_data:
    #         if len(class_type) < max_test_class_number:
    #             GP_augment(class_type, flatten_train_data, max_test_class_number)
    #     save_dataset_multi_input(test_data, 'data/split'+suffix+'_instance0-9/aug_test_data', image=False)

    if save_weight==True:
        train_weight = compute_weight(train_data)
        test_weight = compute_weight(test_data)
        f2 = open('data/split'+suffix+'_instance0-9/class_weights%d'%instance, 'wb')
        pickle.dump((train_weight, test_weight), f2)
        f2.close()
    # f3 = open('data/split'+suffix+'_instance%d/total_dataset'%instance, 'wb')
    # pickle.dump((train_data, val_data, test_data), f3)
    # f3.close()
    return

def GP_augment_exclude_origin(light_curves, flatten_train_data, size):
    '''
    ---
    flatten_train_data : 用于随机抽取观测cadence，以生成模拟观测
    '''
    count = 0
    batch = 0
    initial_data = light_curves.copy()
    augmented_data = []
    error_count = 0
    while count < size:
        batch += 1
        print('augmenting')
        for lc in initial_data:
            x = get_sample_cadence(flatten_train_data, lc.period)
            try:
                simu_lc = lc.generate_GP_simulation(x, phase_shift_ratio=random.uniform(0,1), 
                                                    scale_std=False)
            except:
                error_count += 1
                continue
            simu_lc.class_label = lc.class_label
            simu_lc.id = lc.id + 'generate%d'%batch
            augmented_data.append(lc)
            count += 1
            if count == size:
                break
        print('batch=%d, count=%d'%(batch,count))
    return augmented_data
    
def create_dataset_exclude_origin_multi_test(original_dataset, class_size, aug_val=True, 
                    down_sample=False, down_sample_size=None, instance=1, multi_test=5):
    f1 = open(original_dataset, 'rb')
    train_data, val_data, test_data = pickle.load(f1)
    f1.close()
    flatten_train_data = flat_data(train_data)
    split_file_name = re.match(r'(data/original_dataset)(.*)', original_dataset)
    suffix = split_file_name.group(2) + '_aug_to_%d'%class_size + '_down_sample_%s'%str(down_sample)
    os.mkdir('data/split'+suffix+'_exclude_origin_multi_test_instance%d'%instance)

    if down_sample == True:
        if down_sample_size == None:
            down_sample_size = class_size
        for i, class_type in enumerate(train_data):
            if len(class_type) > down_sample_size:
                train_data[i] = random.sample(class_type, down_sample_size)

    if aug_val == True:
        aug_val_data = []
        max_val_class_number = 1000
        for class_type in val_data:
            aug_val_data.append(GP_augment_exclude_origin(class_type, flatten_train_data, max_val_class_number))
        save_dataset_multi_input(aug_val_data, 'data/split'+suffix+'_exclude_origin_multi_test_instance%d/val_data'%instance)
        print('aug_val finished')
    
    aug_train_data=[]
    for class_type in train_data:
        aug_train_data.append(GP_augment_exclude_origin(class_type, flatten_train_data, class_size))  
    save_dataset_multi_input(aug_train_data, 'data/split'+suffix+'_exclude_origin_multi_test_instance%d/train_data'%instance)
    print('aug_train_finished')

    multi_test_data=[]
    max_test_class_number = 1000
    for i in range(0, multi_test):
        single_test_data = []
        for class_type in test_data:
            single_test_data.append(GP_augment_exclude_origin(class_type, flatten_train_data, max_test_class_number))
        multi_test_data.append(single_test_data)
        
    for i in range(0, multi_test):
        save_dataset_multi_input(multi_test_data[i], 'data/split'+suffix+'_exclude_origin_multi_test_instance%d/test_data%d'%(instance,i))
    print('aug_test_finished')
    return