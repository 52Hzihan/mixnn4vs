import matplotlib.pyplot as plt
import numpy as np
def sequence_to_image(x, y, figsize=(4,4),dpi=32):
    from PIL import Image
    fig = plt.figure(figsize=figsize,dpi=dpi)
    fig.patch.set_facecolor('black')
    plt.scatter(x,y, s=2,c='w')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.margins(0,0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = image.convert('L')
    plt.close('all')
    return image

import pickle
def read_sequence_and_make_image(K=10, fold=0):
    images_test = []
    f2 = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/test_data', 'rb')
    X_sequence_test, X_feature_test, Y_test = pickle.load(f2) 
    f2.close()
    for sequence in X_sequence_test :
        images_test.append(np.array(sequence_to_image(sequence[:,0],sequence[:,1])))
    f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/test_data_image', 'wb')
    pickle.dump(images_test, f)
    f.close()
    print('finish test fold %d'%fold)

    # images_val = []
    # f2 = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/val_data', 'rb')
    # X_sequence_val, X_feature_val, Y_val = pickle.load(f2) 
    # f2.close()
    # for sequence in X_sequence_val :
    #     images_val.append(np.array(sequence_to_image(sequence[:,0],sequence[:,1])))
    # f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/val_data_image', 'wb')
    # pickle.dump(images_val, f)
    # f.close()
    # print('finish val fold %d'%fold)

    # for i in range(0,10):
    #     images_train = []
    #     f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/train_data%d'%i, 'rb')
    #     X_sequence_train, X_feature_train, Y_train = pickle.load(f)
    #     f.close()
    #     for sequence in X_sequence_train :
    #         images_train.append(np.array(sequence_to_image(sequence[:,0],sequence[:,1])))
    #     f = open('data/split_%dfold_No%d'%(K,fold)+'_aug_to_2500_down_sample_True_instance0-9/train_data_image%d'%i, 'wb')
    #     pickle.dump(images_train, f)
    #     f.close()
    #     print('finish train fold %d No.%d'%(fold,i))

for i in range(0,10):
    read_sequence_and_make_image(fold=i)