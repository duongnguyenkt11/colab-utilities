# from __future__ import absolute_import, division, print_function, unicode_literals
import sys , time, os, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras import Model
import tensorflow as tf

if '/tmp' not in sys.path: sys.path.append('/tmp')

class MyLayers:
    def __init__(self):
        # Input & Base
        from tensorflow.python.keras.engine.input_layer import Input
        from tensorflow.python.keras.engine.input_layer import InputLayer
        from tensorflow.python.keras.engine.input_spec import InputSpec
        from tensorflow.python.keras.engine.base_layer import Layer
        self.Input = Input
        self.Layer = Layer

        # Advanced activations.
        from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
        from tensorflow.python.keras.layers.advanced_activations import PReLU
        from tensorflow.python.keras.layers.advanced_activations import ELU
        from tensorflow.python.keras.layers.advanced_activations import ReLU
        from tensorflow.python.keras.layers.advanced_activations import ThresholdedReLU
        from tensorflow.python.keras.layers.advanced_activations import Softmax
        self.ELU = ELU
        self.ReLU = ReLU
        self.softmax = Softmax

        # Convolution layers.
        from tensorflow.python.keras.layers.convolutional import Conv1D
        from tensorflow.python.keras.layers.convolutional import Conv2D
        from tensorflow.python.keras.layers.convolutional import Conv3D
        from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
        from tensorflow.python.keras.layers.convolutional import Conv3DTranspose
        from tensorflow.python.keras.layers.convolutional import SeparableConv1D
        from tensorflow.python.keras.layers.convolutional import SeparableConv2D
        self.Conv2D = Conv2D

        # Convolution layer aliases.
        from tensorflow.python.keras.layers.convolutional import Convolution1D
        from tensorflow.python.keras.layers.convolutional import Convolution2D
        from tensorflow.python.keras.layers.convolutional import Convolution3D
        from tensorflow.python.keras.layers.convolutional import Convolution2DTranspose
        from tensorflow.python.keras.layers.convolutional import Convolution3DTranspose
        from tensorflow.python.keras.layers.convolutional import SeparableConvolution1D
        from tensorflow.python.keras.layers.convolutional import SeparableConvolution2D
        from tensorflow.python.keras.layers.convolutional import DepthwiseConv2D
        # Image processing layers.
        from tensorflow.python.keras.layers.convolutional import UpSampling1D
        from tensorflow.python.keras.layers.convolutional import UpSampling2D
        from tensorflow.python.keras.layers.convolutional import UpSampling3D
        from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
        from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
        from tensorflow.python.keras.layers.convolutional import ZeroPadding3D
        from tensorflow.python.keras.layers.convolutional import Cropping1D
        from tensorflow.python.keras.layers.convolutional import Cropping2D
        from tensorflow.python.keras.layers.convolutional import Cropping3D
        # Core layers.
        from tensorflow.python.keras.layers.core import Masking
        from tensorflow.python.keras.layers.core import Dropout
        from tensorflow.python.keras.layers.core import SpatialDropout1D
        from tensorflow.python.keras.layers.core import SpatialDropout2D
        from tensorflow.python.keras.layers.core import SpatialDropout3D
        from tensorflow.python.keras.layers.core import Activation
        from tensorflow.python.keras.layers.core import Reshape
        from tensorflow.python.keras.layers.core import Permute
        from tensorflow.python.keras.layers.core import Flatten
        from tensorflow.python.keras.layers.core import RepeatVector
        from tensorflow.python.keras.layers.core import Lambda
        from tensorflow.python.keras.layers.core import Dense
        from tensorflow.python.keras.layers.core import ActivityRegularization
        self.Flatten = Flatten
        self.Dense = Dense
        self.Reshape = Reshape
        self.Activation = Activation
        self.Dropout = Dropout

        # Dense Attention layers.
        from tensorflow.python.keras.layers.dense_attention import AdditiveAttention
        from tensorflow.python.keras.layers.dense_attention import Attention

        # Embedding layers.
        from tensorflow.python.keras.layers.embeddings import Embedding
        self.Embedding = Embedding

        # Locally-connected layers.
        from tensorflow.python.keras.layers.local import LocallyConnected1D
        from tensorflow.python.keras.layers.local import LocallyConnected2D

        # Merge layers.
        from tensorflow.python.keras.layers.merge import Add
        from tensorflow.python.keras.layers.merge import Subtract
        from tensorflow.python.keras.layers.merge import Multiply
        from tensorflow.python.keras.layers.merge import Average
        from tensorflow.python.keras.layers.merge import Maximum
        from tensorflow.python.keras.layers.merge import Minimum
        from tensorflow.python.keras.layers.merge import Concatenate
        from tensorflow.python.keras.layers.merge import Dot
        from tensorflow.python.keras.layers.merge import add
        from tensorflow.python.keras.layers.merge import subtract
        from tensorflow.python.keras.layers.merge import multiply
        from tensorflow.python.keras.layers.merge import average
        from tensorflow.python.keras.layers.merge import maximum
        from tensorflow.python.keras.layers.merge import minimum
        from tensorflow.python.keras.layers.merge import concatenate
        from tensorflow.python.keras.layers.merge import dot

        # Noise layers.
        from tensorflow.python.keras.layers.noise import AlphaDropout
        from tensorflow.python.keras.layers.noise import GaussianNoise
        from tensorflow.python.keras.layers.noise import GaussianDropout

        # Normalization layers.
        from tensorflow.python.keras.layers.normalization import LayerNormalization
        from tensorflow.python.keras.layers.normalization import BatchNormalization
        from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization as BatchNormalizationV2
        self.BatchNormalization = BatchNormalization

        # Kernelized layers.
        from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures

        # Pooling layers.
        from tensorflow.python.keras.layers.pooling import MaxPooling1D
        from tensorflow.python.keras.layers.pooling import MaxPooling2D
        from tensorflow.python.keras.layers.pooling import MaxPooling3D
        from tensorflow.python.keras.layers.pooling import AveragePooling1D
        from tensorflow.python.keras.layers.pooling import AveragePooling2D
        from tensorflow.python.keras.layers.pooling import AveragePooling3D
        from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
        from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
        from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPooling3D
        self.MaxPooling2D = MaxPooling2D

        # Pooling layer aliases.
        from tensorflow.python.keras.layers.pooling import MaxPool1D
        from tensorflow.python.keras.layers.pooling import MaxPool2D
        from tensorflow.python.keras.layers.pooling import MaxPool3D
        from tensorflow.python.keras.layers.pooling import AvgPool1D
        from tensorflow.python.keras.layers.pooling import AvgPool2D
        from tensorflow.python.keras.layers.pooling import AvgPool3D
        from tensorflow.python.keras.layers.pooling import GlobalAvgPool1D
        from tensorflow.python.keras.layers.pooling import GlobalAvgPool2D
        from tensorflow.python.keras.layers.pooling import GlobalAvgPool3D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D
        from tensorflow.python.keras.layers.pooling import GlobalMaxPool3D

        # Recurrent layers.
        from tensorflow.python.keras.layers.recurrent import RNN
        from tensorflow.python.keras.layers.recurrent import AbstractRNNCell
        from tensorflow.python.keras.layers.recurrent import StackedRNNCells
        from tensorflow.python.keras.layers.recurrent import SimpleRNNCell
        from tensorflow.python.keras.layers.recurrent import PeepholeLSTMCell
        from tensorflow.python.keras.layers.recurrent import SimpleRNN
        from tensorflow.python.keras.layers.recurrent import GRU
        from tensorflow.python.keras.layers.recurrent import GRUCell
        from tensorflow.python.keras.layers.recurrent import LSTM
        from tensorflow.python.keras.layers.recurrent import LSTMCell
        from tensorflow.python.keras.layers.recurrent_v2 import GRU as GRU_v2
        from tensorflow.python.keras.layers.recurrent_v2 import GRUCell as GRUCell_v2
        from tensorflow.python.keras.layers.recurrent_v2 import LSTM as LSTM_v2
        from tensorflow.python.keras.layers.recurrent_v2 import LSTMCell as LSTMCell_v2
        self.RNN = RNN
        self.LSTM = LSTM

        # Convolutional-recurrent layers.
        from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
        # CuDNN recurrent layers.
        from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNLSTM
        from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU
        # Wrapper functions
        from tensorflow.python.keras.layers.wrappers import Wrapper
        from tensorflow.python.keras.layers.wrappers import Bidirectional
        from tensorflow.python.keras.layers.wrappers import TimeDistributed

        # # RNN Cell wrappers.
        from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DeviceWrapper
        from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import DropoutWrapper
        from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import ResidualWrapper

        # Serialization functions
        from tensorflow.python.keras.layers.serialization import deserialize
        from tensorflow.python.keras.layers.serialization import serialize
class MyMetrics:
    def __init__(self):
        from tensorflow.python.keras.metrics import AUC
        from tensorflow.python.keras.metrics import Accuracy
        from tensorflow.python.keras.metrics import BinaryAccuracy
        from tensorflow.python.keras.metrics import BinaryCrossentropy
        from tensorflow.python.keras.metrics import CategoricalAccuracy
        from tensorflow.python.keras.metrics import CategoricalCrossentropy
        from tensorflow.python.keras.metrics import CategoricalHinge
        from tensorflow.python.keras.metrics import CosineSimilarity
        from tensorflow.python.keras.metrics import FalseNegatives
        from tensorflow.python.keras.metrics import FalsePositives
        from tensorflow.python.keras.metrics import Hinge
        from tensorflow.python.keras.metrics import KLDivergence
        from tensorflow.python.keras.metrics import LogCoshError
        from tensorflow.python.keras.metrics import Mean
        from tensorflow.python.keras.metrics import MeanAbsoluteError
        from tensorflow.python.keras.metrics import MeanAbsolutePercentageError
        from tensorflow.python.keras.metrics import MeanIoU
        from tensorflow.python.keras.metrics import MeanRelativeError
        from tensorflow.python.keras.metrics import MeanSquaredError
        from tensorflow.python.keras.metrics import MeanSquaredLogarithmicError
        from tensorflow.python.keras.metrics import MeanTensor
        from tensorflow.python.keras.metrics import Metric
        from tensorflow.python.keras.metrics import Poisson
        from tensorflow.python.keras.metrics import Precision
        from tensorflow.python.keras.metrics import Recall
        from tensorflow.python.keras.metrics import RootMeanSquaredError
        from tensorflow.python.keras.metrics import SensitivityAtSpecificity
        from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
        from tensorflow.python.keras.metrics import SparseCategoricalCrossentropy
        from tensorflow.python.keras.metrics import SparseTopKCategoricalAccuracy
        from tensorflow.python.keras.metrics import SpecificityAtSensitivity
        from tensorflow.python.keras.metrics import SquaredHinge
        from tensorflow.python.keras.metrics import Sum
        from tensorflow.python.keras.metrics import TopKCategoricalAccuracy
        from tensorflow.python.keras.metrics import TrueNegatives
        from tensorflow.python.keras.metrics import TruePositives
        from tensorflow.python.keras.metrics import binary_accuracy
        from tensorflow.python.keras.metrics import categorical_accuracy
        from tensorflow.python.keras.metrics import deserialize
        from tensorflow.python.keras.metrics import get
        from tensorflow.python.keras.metrics import serialize
        from tensorflow.python.keras.metrics import sparse_categorical_accuracy
        from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy
        from tensorflow.python.keras.metrics import top_k_categorical_accuracy

        self.Accuracy = Accuracy
        self.BinaryAccuracy = BinaryAccuracy
        self.BinaryCrossentropy = BinaryCrossentropy
        self.CategoricalAccuracy = CategoricalAccuracy
        self.CategoricalCrossentropy = CategoricalCrossentropy
        self.CategoricalHinge = CategoricalHinge
        self.CosineSimilarity = CosineSimilarity
        self.FalseNegatives = FalseNegatives
        self.FalsePositives = FalsePositives
        self.Hinge = Hinge
        self.KLDivergence = KLDivergence
        self.LogCoshError = LogCoshError
        self.Mean = Mean
        self.MeanAbsoluteError = MeanAbsoluteError
        self.MeanAbsolutePercentageError = MeanAbsolutePercentageError
        self.MeanIoU = MeanIoU
        self.MeanRelativeError = MeanRelativeError
        self.MeanSquaredError = MeanSquaredError
        self.MeanSquaredLogarithmicError = MeanSquaredLogarithmicError
        self.MeanTensor = MeanTensor
        self.Metric = Metric
        self.Poisson = Poisson
        self.Precision = Precision
        self.Recall = Recall
        self.RootMeanSquaredError = RootMeanSquaredError
        self.SensitivityAtSpecificity = SensitivityAtSpecificity
        self.SparseCategoricalAccuracy = SparseCategoricalAccuracy
        self.SparseCategoricalCrossentropy = SparseCategoricalCrossentropy
        self.SparseTopKCategoricalAccuracy = SparseTopKCategoricalAccuracy
        self.SpecificityAtSensitivity = SpecificityAtSensitivity
        self.SquaredHinge = SquaredHinge
        self.Sum = Sum
        self.TopKCategoricalAccuracy = TopKCategoricalAccuracy
        self.TrueNegatives = TrueNegatives
        self.TruePositives = TruePositives
        self.binary_accuracy = binary_accuracy
        self.categorical_accuracy = categorical_accuracy
        self.deserialize = deserialize
        self.get = get
        self.serialize = serialize
        self.sparse_categorical_accuracy = sparse_categorical_accuracy
        self.sparse_top_k_categorical_accuracy = sparse_top_k_categorical_accuracy
        self.top_k_categorical_accuracy = top_k_categorical_accuracy
class MyLosses:
    def __init__(self):
        from tensorflow.python.keras.losses import BinaryCrossentropy
        from tensorflow.python.keras.losses import CategoricalCrossentropy
        from tensorflow.python.keras.losses import CategoricalHinge
        from tensorflow.python.keras.losses import CosineSimilarity
        from tensorflow.python.keras.losses import Hinge
        from tensorflow.python.keras.losses import Huber
        from tensorflow.python.keras.losses import KLD
        from tensorflow.python.keras.losses import KLD as kld
        from tensorflow.python.keras.losses import KLD as kullback_leibler_divergence
        from tensorflow.python.keras.losses import KLDivergence
        from tensorflow.python.keras.losses import LogCosh
        from tensorflow.python.keras.losses import Loss
        from tensorflow.python.keras.losses import MAE
        from tensorflow.python.keras.losses import MAE as mae
        from tensorflow.python.keras.losses import MAE as mean_absolute_error
        from tensorflow.python.keras.losses import MAPE
        from tensorflow.python.keras.losses import MAPE as mape
        from tensorflow.python.keras.losses import MAPE as mean_absolute_percentage_error
        from tensorflow.python.keras.losses import MSE
        from tensorflow.python.keras.losses import MSE as mean_squared_error
        from tensorflow.python.keras.losses import MSE as mse
        from tensorflow.python.keras.losses import MSLE
        from tensorflow.python.keras.losses import MSLE as mean_squared_logarithmic_error
        from tensorflow.python.keras.losses import MSLE as msle
        from tensorflow.python.keras.losses import MeanAbsoluteError
        from tensorflow.python.keras.losses import MeanAbsolutePercentageError
        from tensorflow.python.keras.losses import MeanSquaredError
        from tensorflow.python.keras.losses import MeanSquaredLogarithmicError
        from tensorflow.python.keras.losses import Poisson
        from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
        from tensorflow.python.keras.losses import SquaredHinge
        from tensorflow.python.keras.losses import binary_crossentropy
        from tensorflow.python.keras.losses import categorical_crossentropy
        from tensorflow.python.keras.losses import categorical_hinge
        from tensorflow.python.keras.losses import cosine_similarity
        from tensorflow.python.keras.losses import deserialize
        from tensorflow.python.keras.losses import get
        from tensorflow.python.keras.losses import hinge
        from tensorflow.python.keras.losses import logcosh
        from tensorflow.python.keras.losses import poisson
        from tensorflow.python.keras.losses import serialize
        from tensorflow.python.keras.losses import sparse_categorical_crossentropy
        from tensorflow.python.keras.losses import squared_hinge

        self.BinaryCrossentropy = BinaryCrossentropy
        self.CategoricalCrossentropy = CategoricalCrossentropy
        self.CategoricalHinge = CategoricalHinge
        self.CosineSimilarity = CosineSimilarity
        self.Hinge = Hinge
        self.Huber = Huber
        self.KLD = KLD
        self.kullback_leibler_divergence = kullback_leibler_divergence
        self.kld = kld
        self.KLDivergence = KLDivergence
        self.LogCosh = LogCosh
        self.Loss = Loss
        self.MAE = MAE
        self.mae = mae
        self.mean_absolute_error = mean_absolute_error
        self.MAPE = MAPE
        self.mape = mape
        self.mean_absolute_percentage_error = mean_absolute_percentage_error
        self.MSE = MSE
        self.mean_squared_error = mean_squared_error
        self.mse = mse
        self.MSLE = MSLE
        self.msle = msle
        self.mean_squared_logarithmic_error = mean_squared_logarithmic_error
        self.MeanAbsoluteError = MeanAbsoluteError
        self.MeanAbsolutePercentageError = MeanAbsolutePercentageError
        self.MeanSquaredError = MeanSquaredError
        self.MeanSquaredLogarithmicError = MeanSquaredLogarithmicError
        self.Poisson = Poisson
        self.SparseCategoricalCrossentropy = SparseCategoricalCrossentropy
        self.SquaredHinge = SquaredHinge
        self.binary_crossentropy = binary_crossentropy
        self.categorical_crossentropy = categorical_crossentropy
        self.categorical_hinge = categorical_hinge
        self.cosine_similarity = cosine_similarity
        self.deserialize = deserialize
        self.get = get
        self.hinge = hinge
        self.logcosh = logcosh
        self.poisson = poisson
        self.serialize = serialize
        self.sparse_categorical_crossentropy = sparse_categorical_crossentropy
        self.squared_hinge = squared_hinge

Losses = MyLosses()
Metrics = MyMetrics()
L = MyLayers()
