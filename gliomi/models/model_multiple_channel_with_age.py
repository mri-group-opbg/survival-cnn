from tensorflow.keras.models import Model

# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.layers import Concatenate, Dense

class CNNCombinedModel(Model):
    
    def __init__(self, cnns):
        inputs, outputs = self.combined_model(cnns)
        super().__init__(inputs, outputs)

    def combined_model(self, cnns, regularize=False):
        
        cnn_outpus = []
        cnn_inputs = []
        for cnn in cnns:
            cnn_outpus.append(cnn.output)
            cnn_inputs.append(cnn.input)

        combinedInput = Concatenate()(cnn_outpus)

        # our final FC layer head will have two dense layers, the final one
        # being our regression head
        X = Dense(256, activation="relu")(combinedInput)
        X = Dense(2, activation="softmax")(X)

        return cnn_inputs, X
    