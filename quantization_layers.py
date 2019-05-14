
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

def quantize(x, bits, scale, signed, fp=True, floor=False):
    '''
    Quantizes tensor x.
    The returned value will be x quantized to `bits` bits ranging from -scale to scale (if signed) or 0 to scale (if unsigned).
    `fp` is used to bypass quantization so that the same network can represent a floating point or a quantized network.
    '''
    if fp: return x + 0*scale
    midrise = signed and (bits <= 2)
    qmax = 2**(bits-1) if signed else 2**bits
    s = x * qmax / scale
    rounded = tf.floor(s) if floor else (tf.floor(s)+0.5 if midrise else tf.round(s))
    return tf.clip_by_value(s + tf.stop_gradient(rounded - s), -qmax + midrise*0.5 if signed else 0, qmax - 1 + midrise*0.5) * scale / qmax

def log_thresh_to_thresh(lt, gradient_mult=0.0):
    '''
    Quantization scales are stored as logs of the actual desired scales.
    This function converts back by exponentiating.
    '''
    glt = gradient_mult * lt
    t_smooth = K.pow(2.0, glt + tf.stop_gradient(K.round(lt) - glt))
    return t_smooth

class QuantQ(Layer):
    '''
    Simple quantization layer that has a settable quantization scale.
    '''
    def __init__(self, aq=4, fp=True, qt=0, **kwargs):
        self.aq = aq
        self.fp = fp
        self.qt = qt
        super(QuantQ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alt = self.add_weight(name='a_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(QuantQ, self).build(input_shape)

    def call(self, x):
        return quantize(x, self.aq, log_thresh_to_thresh(self.alt, self.qt), True, fp=self.fp)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            'aq': self.aq,
            'fp': self.fp,
            'qt': self.qt,
        }
        base_config = super(QuantQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DenseQ(Layer):
    '''
    Dense layer with quantization (allows different weight/bias/activation quantizations).
    '''
    def __init__(self, output_dim, activation=None, wq=4, bq=8, aq=4, fp=True, qt=0, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.wq = wq
        self.bq = bq
        self.aq = aq
        self.fp = fp
        self.qt = qt
        super(DenseQ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(input_shape[1], self.output_dim), initializer='orthogonal', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.wlt = self.add_weight(name='w_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.blt = self.add_weight(name='b_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.alt = self.add_weight(name='a_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(DenseQ, self).build(input_shape)

    def call(self, x):
        z = K.dot(x, quantize(self.kernel, self.wq, log_thresh_to_thresh(self.wlt, self.qt), True, fp=self.fp))
        z += quantize(self.bias, self.bq, log_thresh_to_thresh(self.blt, self.qt), True, fp=self.fp)
        if self.activation: return quantize(self.activation(z), self.aq, log_thresh_to_thresh(self.alt, self.qt), False, fp=self.fp)
        return quantize(z, self.aq, log_thresh_to_thresh(self.alt, self.qt), True, fp=self.fp)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'wq': self.wq,
            'bq': self.bq,
            'aq': self.aq,
            'fp': self.fp,
            'qt': self.qt,
        }
        base_config = super(DenseQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvQ(Layer):
    '''
    Convolution layer with quantization (allows different weight/bias/activation quantizations).
    '''
    def __init__(self, output_dim, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, wq=4, bq=8, aq=4, fp=True, qt=0, **kwargs):
        self.output_dim = output_dim
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding
        self.activation = activations.get(activation)
        self.wq = wq
        self.bq = bq
        self.aq = aq
        self.fp = fp
        self.qt = qt
        super(ConvQ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.output_dim), initializer='orthogonal', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.wlt = self.add_weight(name='w_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.blt = self.add_weight(name='b_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.alt = self.add_weight(name='a_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(ConvQ, self).build(input_shape)

    def call(self, x):
        z = K.conv2d(x, quantize(self.kernel, self.wq, log_thresh_to_thresh(self.wlt, self.qt), True, fp=self.fp), strides=self.strides, padding=self.padding)
        z += quantize(self.bias, self.bq, log_thresh_to_thresh(self.blt, self.qt), True, fp=self.fp)
        if self.activation: return quantize(self.activation(z), self.aq, log_thresh_to_thresh(self.alt, self.qt), False, fp=self.fp)
        return quantize(z, self.aq, log_thresh_to_thresh(self.alt, self.qt), True, fp=self.fp)
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            pad = 0 if self.padding == 'valid' else (self.kernel_size[i] - 1) // 2
            new_dim = (space[i] - self.kernel_size[i] + 2*pad) // self.strides[i] + 1
            new_space.append(new_dim)
        out_shape = (input_shape[0],) + tuple(new_space) + (self.output_dim,)
        return out_shape
    
    def get_config(self):
        config = {
            'output_dim' : self.output_dim,
            'kernel_size': self.kernel_size,
            'strides'    : self.strides,
            'padding'    : self.padding,
            'activation' : activations.serialize(self.activation),
            'wq': self.wq,
            'bq': self.bq,
            'aq': self.aq,
            'fp': self.fp,
            'qt': self.qt,
        }
        base_config = super(ConvQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ThinConvQ(Layer):
    '''
    Thin convolution layer with quantization (allows different weight/bias/activation quantizations).
    Use with 1x1 convolution for depthwise-separable convolutions.
    '''
    def __init__(self, multiplier, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, wq=4, bq=8, aq=4, fp=True, qt=0, **kwargs):
        self.multiplier = multiplier
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding
        self.activation = activations.get(activation)
        self.wq = wq
        self.bq = bq
        self.aq = aq
        self.fp = fp
        self.qt = qt
        super(ThinConvQ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.kernel_size + (input_shape[-1], self.multiplier), initializer='orthogonal', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(input_shape[-1]*self.multiplier,), initializer='zeros', trainable=True)
        self.wlt = self.add_weight(name='w_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.blt = self.add_weight(name='b_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.alt = self.add_weight(name='a_log_threshold', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(ThinConvQ, self).build(input_shape)

    def call(self, x):
        z = K.depthwise_conv2d(x, quantize(self.kernel, self.wq, log_thresh_to_thresh(self.wlt, self.qt), True, fp=self.fp), strides=self.strides, padding=self.padding)
        z += quantize(self.bias, self.bq, log_thresh_to_thresh(self.blt, self.qt), True, fp=self.fp)
        if self.activation: return quantize(self.activation(z), self.aq, log_thresh_to_thresh(self.alt, self.qt), False, fp=self.fp)
        return quantize(z, self.aq, log_thresh_to_thresh(self.alt, self.qt), True, fp=self.fp)
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            pad = 0 if self.padding == 'valid' else (self.kernel_size[i] - 1) // 2
            new_dim = (space[i] - self.kernel_size[i] + 2*pad) // self.strides[i] + 1
            new_space.append(new_dim)
        out_shape = (input_shape[0],) + tuple(new_space) + (input_shape[-1]*self.multiplier,)
        return out_shape
    
    def get_config(self):
        config = {
            'multiplier' : self.multiplier,
            'kernel_size': self.kernel_size,
            'strides'    : self.strides,
            'padding'    : self.padding,
            'activation' : activations.serialize(self.activation),
            'wq': self.wq,
            'bq': self.bq,
            'aq': self.aq,
            'fp': self.fp,
            'qt': self.qt,
        }
        base_config = super(ThinConvQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ResidQ(Layer):
    '''
    Residual connection block with quantization.
    Allows different weight/bias/activation quantizations.
    Has settable scales for the six separate quantized components (weight, bias, activations x 2).
    '''
    def __init__(self, wq=4, bq=8, aq=4, fp=True, qt=0, **kwargs):
        self.kernel_size = (3,3)
        self.strides = (1,1)
        self.padding = 'same'
        self.activation = activations.get('relu')
        self.wq = wq
        self.bq = bq
        self.aq = aq
        self.fp = fp
        self.qt = qt
        super(ResidQ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        self.kernel1 = self.add_weight(name='kernel1', shape=self.kernel_size + (input_shape[-1], self.output_dim), initializer='orthogonal', trainable=True)
        self.bias1   = self.add_weight(name='bias1', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.kernel2 = self.add_weight(name='kernel2', shape=self.kernel_size + (input_shape[-1], self.output_dim), initializer='orthogonal', trainable=True)
        self.bias2   = self.add_weight(name='bias2', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.wlt1 = self.add_weight(name='w_log_threshold1', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.blt1 = self.add_weight(name='b_log_threshold1', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.alt1 = self.add_weight(name='a_log_threshold1', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.wlt2 = self.add_weight(name='w_log_threshold2', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.blt2 = self.add_weight(name='b_log_threshold2', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        self.alt2 = self.add_weight(name='a_log_threshold2', shape=(), initializer=keras.initializers.Constant(value=0), trainable=True)
        super(ResidQ, self).build(input_shape)

    def call(self, x):
        xin = x
        z1 = K.conv2d(x, quantize(self.kernel1, self.wq, log_thresh_to_thresh(self.wlt1, self.qt), True, fp=self.fp), strides=self.strides, padding=self.padding)
        z1 += quantize(self.bias1, self.bq, log_thresh_to_thresh(self.blt1, self.qt), True, fp=self.fp)
        y1 = quantize(self.activation(z1), self.aq, log_thresh_to_thresh(self.alt1, self.qt), False, fp=self.fp)
        z2 = K.conv2d(y1, quantize(self.kernel2, self.wq, log_thresh_to_thresh(self.wlt2, self.qt), True, fp=self.fp), strides=self.strides, padding=self.padding)
        z2 += quantize(self.bias2, self.bq, log_thresh_to_thresh(self.blt2, self.qt), True, fp=self.fp)
        z2 += xin
        return quantize(self.activation(z2), self.aq, log_thresh_to_thresh(self.alt2, self.qt), False, fp=self.fp)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            'wq': self.wq,
            'bq': self.bq,
            'aq': self.aq,
            'fp': self.fp,
            'qt': self.qt,
        }
        base_config = super(ResidQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def percentile_limit(x, pct):
    '''
    For the given tensor, x, gives the values at the pct percentile and 100-pct percentile.
    This is used to come up with reasonable scales for quantization in loadQ.
    '''
    lim_l = np.percentile(x, pct)
    lim_h = np.percentile(x, 100-pct)
    return np.max(np.abs([lim_l, lim_h]))

def loadQ(model, modelQ, Xva, pct=2, verbose=False):
    '''
    Loads the weights from model to modelQ and sets the quantization scales.
    Requires a small sample of data to derive the activations distributions.
    '''
    mq = []
    qi = 0
    for i in range(len(modelQ.layers)):
        if len(modelQ.layers[i].get_weights()) >= 2: mq.append(i)
    acts = K.function([model.layers[0].input, K.learning_phase()], map(lambda x: x.output, model.layers)[1:])([Xva[:128], 0])
    for i in range(len(model.layers)):
        layer = model.layers[i]
        ws = layer.get_weights()
        if len(ws) < 2: continue
        wt = percentile_limit(ws[0], pct)
        bt = 2 * (0.1 + np.max(np.abs(ws[1])))
        act = acts[i-1].flatten()
        if np.any(act<0): at = percentile_limit(act, pct)
        else: at = percentile_limit(act[act>0], pct)
        setw = [ws[0], ws[1], np.log2(wt), np.log2(bt), np.log2(at)]
        if len(ws) >= 10:
            wt = percentile_limit(ws[2], pct)
            bt = 2 * (0.1 + np.max(np.abs(ws[3])))
            setw = setw[:2] + [ws[2], ws[3]] + setw[2:] + [np.log2(wt), np.log2(bt), np.log2(at)]
        modelQ.layers[mq[qi]].set_weights(setw)
        qi += 1
        if verbose:
            print('Quantization layer %d: w(%.4f +/- %.4f) b(%.4f +/- %.4f) qw(%.4f) qb(%.4f) qa(%.4f)'%(
                i, np.mean(ws[0]), np.std(ws[0]), np.mean(ws[1]), np.std(ws[1]), wt, bt, at))

