
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.utils as KU
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Activation, GaussianNoise, Reshape
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine.topology import Layer
from keras import regularizers, activations

from quantization_layers import *

def compute_conv_slack(ca, cb, ksize=(3,3), strides=(1,1), convolution_strategy='naive'):
    '''
    Computes the extra (beyond input feature map storage) memory required to compute
    a convolution with the provided input/output shapes and kernel size / stride.
    You can try different convolution strategies to see how whether a design close
    to meeting the memory constraint can be improved enough by using a more memory
    optimal strategy.
    '''
    ha, wa, fa = ca
    hb, wb, fb = cb
    kr, kc = ksize
    sr, sc = strides
    if convolution_strategy == 'naive': # allocate all the memory for new activations
        return (hb//sr) * (wb//sc) * fb
    if fb <= fa * sr * sc: return 0
    if sr != 1 or sc != 1:
        # Strides not analyzed for complicated cases. Assume "normal" order for these cases.
        return max(0, hb * wb * fb - ha * wa * fa) + kc * fa
    if wb + (kr - kc) * fa / (fb - fa) > hb:
        ha, wa, hb, wb, kr, kc, sr, sc = wa, ha, wb, hb, kc, kr, sc, sr # transpose image
    if convolution_strategy == 'normal': # row-by-row convolutions
        return hb * (wb * fb - wa * fa) + kc * fa
    if convolution_strategy == 'transpose': # one transpose in the middle (and possibly one at the beginning)
        x0 = int((kr-1)*fa / (fb-fa))
        t1 = (hb-x0) * (wb*(fb-fa) - (kc-1)*fa) + kc*fa
        x0 += 1
        t2 = (hb-x0) * (wb*(fb-fa) - (kc-1)*fa) + wb*(x0*(fb-fa) - (kr-1)*fa) + kr*fa
        cond = ((kr-1)*fa % (fb-fa) > (kr-1)*fa/wb) == (t2 < t1)
        if not cond:
            print('Condition not satisfied.')
        return min(t1, t2)
    if convolution_strategy == 'herringbone': # transpose after each row/col
        hc, wc = hb, wb
        debt = 0
        last = 0
        while wc > 0 and hc > 0:
            if wc + (kr - kc) * fa / (fb - fa) > hc:
                last = kr * fa
                debt += max(0, hc * (fb - fa) - (kr - 1) * fa)
                wc -= 1
            else:
                last = kc * fa
                debt += max(0, wc * (fb - fa) - (kc - 1) * fa)
                hc -= 1
        return debt + last

def compute_storage(nn, input_dimensions=[28,28,1], input_bits=8, streaming='true', convolution_strategy='herringbone', verbose=False):
    '''
    Computes the overall storage requirements of the given network paramterization.
    Returns separately the weight storage costs and the activation storage costs.
    '''
    cur = input_dimensions
    tx  = input_bits/8
    store_w = 0
    store_l = 0 if streaming else np.product(cur)*tx
    store_a = store_l
    for l in nn:
        if l[0] == 'A':
            t, k, ab = l
            ab /= 8
            cur = [ (cur[0]+k-1)//k, (cur[1]+k-1)//k, cur[2] ]
            store_l = np.product(cur) * ab
            store_a = max(store_a, store_l)
        if l[0] == 'C' or l[0] == 'CN':
            if len(l) > 6:
                t, do, kr, kc, sr, sc, wb, bb, ab = l
            else:
                t, do, k, wb, bb, ab = l
                kr = k
                kc = k
                sr = 1
                sc = 1
            di = cur[-1]
            wb /= 8; bb /= 8; ab /= 8
            store_w += kr*kc*di*do*wb + do*bb
            cur_new = [ (cur[0]-kr)//sr+1, (cur[1]-kc)//sc+1, do ]
            store_l = compute_conv_slack(cur, cur_new, (kr,kc), (sr,sc), convolution_strategy)*ab + np.product(cur)*tx
            cur = cur_new
            store_a = max(store_a, store_l)
        if l[0] == 'T':
            if len(l) > 6:
                t, m, kr, kc, sr, sc, wb, bb, ab = l
            else:
                t, m, k, wb, bb, ab = l
                kr = k
                kc = k
                sr = 1
                sc = 1
            di = cur[-1]
            do = di * m
            wb /= 8; bb /= 8; ab /= 8
            store_w += kr*kc*di*m*wb + do*bb
            cur_new = [ (cur[0]-kr)//sr+1, (cur[1]-kc)//sc+1, do ]
            store_l = max(np.product(cur)*tx, np.product(cur_new)*ab)
            cur = cur_new
            store_a = max(store_a, store_l)
        if l[0] == 'DSC':
            if len(l) > 6:
                t, do, kr, kc, sr, sc, wb, bb, ab = l
            else:
                t, do, k, wb, bb, ab = l
                kr = k
                kc = k
                sr = 1
                sc = 1
            di = cur[-1]
            wb /= 8; bb /= 8; ab /= 8
            store_w += kr*kc*di*1*wb + di*1*bb + 1*1*di*do*wb + do*bb
            cur_new = [ (cur[0]-kr)//sr+1, (cur[1]-kc)//sc+1, do ]
            store_l = max(np.product(cur)*tx, np.product(cur_new)*ab)
            cur = cur_new
            store_a = max(store_a, store_l)
        if l[0] == 'R':
            t, wb, bb, ab = l
            wb /= 8; bb /= 8; ab /= 8
            di, do = cur[-1], cur[-1]
            store_w += 2*3*3*di*do*wb + 2*do*bb
            store_l = (3*cur[1]+2)*ab + np.product(cur)*tx
            store_a = max(store_a, store_l)
        if l[0] == 'M':
            t, k, ab = l
            ab /= 8
            cur = [ (cur[0]+k-1)//k, (cur[1]+k-1)//k, cur[2] ]
            store_l = np.product(cur) * ab
            store_a = max(store_a, store_l)
        if l[0] == 'F' or l[0] == 'S':
            t, do, wb, bb, ab = l
            di = np.product(cur)
            wb /= 8; bb /= 8; ab /= 8
            store_w += di*do*wb + do*bb
            store_l = do*ab + np.ceil(np.log2(di))*2*wb + di*tx
            cur = [ do ]
            store_a = max(store_a, store_l)
        if l[0] == 'Q':
            pass
        if l[0] == 'N':
            pass
        if l[0] == 'D':
            pass
        tx = l[-1]/8
        if verbose: print(cur, store_w, store_a)
        if np.any(np.array(cur) <= 0):
            return -1, -1
    return store_w, store_a

def output_logits(X, config, fp=True, qt=0):
    '''
    Generates the Keras network from the provided paramterization config.
    '''
    for l in config:
        if l[0] == 'Q':
            X = QuantQ(aq=l[1], fp=fp, qt=qt)(X)
        if l[0] == 'A':
            X = AveragePooling2D(pool_size=(l[1], l[1]))(X)
            X = Lambda(lambda x: quantize(x, l[2], 1.0, False, fp, floor=True))(X)
        if l[0] == 'C':
            X = ConvQ(l[1], (l[2], l[3]), strides=(l[4], l[5]), activation='relu', wq=l[6], bq=l[7], aq=l[8], fp=fp, qt=qt)(X)
        if l[0] == 'CN':
            X = ConvQ(l[1], (l[2], l[3]), strides=(l[4], l[5]), activation=None, wq=l[6], bq=l[7], aq=l[8], fp=fp, qt=qt)(X)
        if l[0] == 'DSC':
            X = ThinConvQ(1, (l[2], l[3]), strides=(l[4], l[5]), activation=None, wq=l[6], bq=l[7], aq=l[8], fp=fp, qt=qt)(X)
            X = ConvQ(l[1], (1, 1), strides=(1, 1), activation='relu', wq=l[6], bq=l[7], aq=l[8], fp=fp, qt=qt)(X)
        if l[0] == 'T':
            X = ThinConvQ(l[1], (l[2], l[3]), strides=(l[4], l[5]), activation=None, wq=l[6], bq=l[7], aq=l[8], fp=fp, qt=qt)(X)
        if l[0] == 'R':
            X = ResidQ(wq=l[1], bq=l[2], aq=l[3], fp=fp, qt=qt)(X)
        if l[0] == 'M':
            X = MaxPooling2D(pool_size=(l[1], l[1]))(X)
        if l[0] == 'F':
            if len(X.shape) > 2: X = Flatten()(X)
            X = DenseQ(l[1], activation='relu', wq=l[2], bq=l[3], aq=l[4], fp=fp, qt=qt)(X)
        if l[0] == 'N':
            X = GaussianNoise(l[1])(X)
            X = QuantQ(aq=l[2], fp=fp, qt=qt)(X)
        if l[0] == 'D':
            X = Dropout(l[1])(X)
        if l[0] == 'S':
            if len(X.shape) > 2: X = Flatten()(X)
            X = DenseQ(l[1], activation=None, wq=l[2], bq=l[3], aq=l[4], fp=fp, qt=qt)(X)
    return X
