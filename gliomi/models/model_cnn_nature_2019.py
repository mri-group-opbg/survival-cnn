"""
Article of Nature... TBD
"""

from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

class GliomiNature2019(Model):
    
    def __init__(self, n_features, input_shape=(224,224,1), classes=2, zero_pad=True, avg_pool=True, random_seed=42):
        inputs, outputs = self.CNN_2D(n_features, input_shape, classes, zero_pad, avg_pool, random_seed)
        super().__init__(inputs, outputs)

    def ConvN(self, X, filters, stride, batch_norm=True, max_pool=True, random_seed=42):

        for i, f in enumerate(filters):

            X = Conv2D(filters = f, 
                       kernel_size = (3, 3), 
                       strides = (stride, stride), 
                       padding = 'valid', 
                       kernel_initializer = glorot_uniform(seed=random_seed))(X)

            X = Activation('relu')(X)

            if batch_norm:
                X = BatchNormalization(axis = 3)(X)

        if max_pool:
            X = MaxPooling2D((2, 2), strides=(stride, stride))(X)

        return X

    def CNN_2D(self, n_features, input_shape=(224,224,1), classes=2, zero_pad=True, avg_pool=True, random_seed=42):

        X = Input(input_shape)
        X_input = X

        if zero_pad:
            X = ZeroPadding2D((3, 3))(X)

        X = self.ConvN(X, [64,128,256], stride=1, batch_norm=True, max_pool=True, random_seed=random_seed)
        X = self.ConvN(X, [64,128], stride=1, batch_norm=True, max_pool=True, random_seed=random_seed)
        X = self.ConvN(X, [64], stride=2, batch_norm=True, max_pool=True, random_seed=random_seed)
        X = self.ConvN(X, [64], stride=2, batch_norm=True, max_pool=False, random_seed=random_seed)

        if avg_pool:
            X = AveragePooling2D((2,2))(X)

        X = Flatten()(X)

        X = Dense(n_features, 
                  activation='relu', 
                  kernel_initializer = glorot_uniform(seed=random_seed))(X)

        X = Dense(classes, 
                  activation='softmax', 
                  kernel_initializer = glorot_uniform(seed=random_seed))(X)

        return X_input, X