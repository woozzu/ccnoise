import numpy as np
from scipy.io import loadmat
from os import mkdir

from skimage import feature
from skimage.color import rgb2hsv
from skimage.morphology import dilation, square

import caffe
import lmdb

statlist = np.array(['data/stats.mat'])

ti_env = lmdb.open('data/train_input', map_size=1024000000000)
tl_env = lmdb.open('data/train_label', map_size=1024000000000)
vi_env = lmdb.open('data/val_input', map_size=1024000000000)
vl_env = lmdb.open('data/val_label', map_size=1024000000000)

tcount = 0
vcount = 0

for dset in xrange(statlist.shape[0]):
    print 'Loading mat data...'.format(dset)

    stats = loadmat(statlist[dset])
    img_cov = stats['img_cov'].astype('float')
    img_mean = stats['img_mean'].astype('float')
    img_mask = stats['img_mask'].astype('bool')

    selected = np.zeros((img_mean.shape[0], img_mean.shape[1]))
    sp = np.zeros([img_mean.shape[0] / 8, img_mean.shape[1] / 8, img_mean.shape[2]])
    sp_mask = np.zeros([img_mean.shape[0] / 8, img_mean.shape[1] / 8], dtype='bool')

    print 'Generating dataset...'
    
    # Superpixel 8*8
    for i in xrange(0, img_mean.shape[0], 8):
        for j in xrange(0, img_mean.shape[1], 8):
            sp[i / 8, j / 8, :] = np.mean(np.mean(img_mean[i:i+8, j:j+8, :], axis=0), axis=0)
            sp_mask[i / 8, j / 8] = img_mask[i, j]

    # Edge detection
    gray = rgb2hsv(sp)[:, :, 2]
    edge = feature.canny(gray, sigma=3)

    # Dilation
    edge = dilation(edge, square(3))
    edge = edge & ~sp_mask

    for i in xrange(0, img_mean.shape[0], 8):
        for j in xrange(0, img_mean.shape[1], 8):
            if edge[i / 8, j / 8]:
                selected[i:i+8, j:j+8] = 1

    idx = np.where(selected == 1)
    perms = np.random.permutation(idx[0].shape[0])
    split = np.floor(idx[0].shape[0] * 0.98).astype('int')

    print '{0} samples were generated (98%: training set, 2%: validation set).'.format(idx[0].shape[0])
    print 'Saving...'

    with ti_env.begin(write=True) as ti_txn, tl_env.begin(write=True) as tl_txn:
        for i in xrange(split):
            r = idx[0][perms[i]]
            c = idx[1][perms[i]]
            patch = img_mean[r - r % 8:r - r % 8 + 8, c - c % 8:c - c % 8 + 8, :].reshape(192)
            color = img_mean[r, c, :].reshape(3)
            d_input = caffe.io.array_to_datum(np.concatenate((color, patch)).reshape((195, 1, 1)))
            d_label = caffe.io.array_to_datum(img_cov[r, c, :].reshape((6, 1, 1)))
            ti_txn.put('{:08}'.format(tcount), d_input.SerializeToString())
            tl_txn.put('{:08}'.format(tcount), d_label.SerializeToString())
            tcount = tcount + 1

    with vi_env.begin(write=True) as vi_txn, vl_env.begin(write=True) as vl_txn:
        for i in xrange(split + 1, idx[0].shape[0]):
            r = idx[0][perms[i]]
            c = idx[1][perms[i]]
            patch = img_mean[r - r % 8:r - r % 8 + 8, c - c % 8:c - c % 8 + 8, :].reshape(192)
            color = img_mean[r, c, :].reshape(3)
            d_input = caffe.io.array_to_datum(np.concatenate((color, patch)).reshape((195, 1, 1)))
            d_label = caffe.io.array_to_datum(img_cov[r, c, :].reshape((6, 1, 1)))
            vi_txn.put('{:08}'.format(vcount), d_input.SerializeToString())
            vl_txn.put('{:08}'.format(vcount), d_label.SerializeToString())
            vcount = vcount + 1

    print 'Done.'

ti_env.close()
tl_env.close()
vi_env.close()
vl_env.close()
