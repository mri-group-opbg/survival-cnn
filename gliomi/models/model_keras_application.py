from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Input

# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from tensorflow.keras.layers import Concatenate, Dense

class KerasApplicationClassifier(Model):

    def __init__(self, model_name, num_features=2, include_dropout=False, input_shape=(224, 224, 3)):
        inputs, outputs = self.keras_application_classifier(model_name, num_features=num_features, include_dropout=include_dropout, input_shape=input_shape)
        super().__init__(inputs, outputs)
        
    def keras_application_classifier(self, model_name, num_features=2, include_dropout=False, input_shape=(224, 224, 3)):
        
        # Base model is a Keras Application
        base_model = eval(model_name)(weights=None, include_top=False, input_tensor=Input(input_shape))

        # Pooling layer
        pool_layer = AveragePooling2D(pool_size=(4, 4))(base_model.output)

        # Flatten before dense layers
        flatten_layer = Flatten()(pool_layer)

        # Feature leayer
        if include_dropout:
            # Drop-out
            feature_layer = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(flatten_layer)
            feature_layer = Dropout(0.5)(feature_layer)
        else:
            feature_layer = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(flatten_layer)

        # and a logistic layer -- let's say we have 200 classes
        prediction_layer = Dense(2, activation='softmax', kernel_initializer='glorot_uniform')(feature_layer)
        
        return base_model.input, prediction_layer