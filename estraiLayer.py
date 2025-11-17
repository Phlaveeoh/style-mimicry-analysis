from tensorflow import keras

#modelli scelti
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.applications.convnext import ConvNeXtBase

from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import visualkeras

def estraiUltimoLayer(model):
    ultimo_layer = model.layers[-2]
    feature_model = Model(inputs=model.input, outputs=ultimo_layer.output)
    return feature_model

def estraiUltimoMultidimensionale(model):
    for layer in reversed(model.layers):
        shape = getattr(layer.output, 'shape', None)
        if shape is not None and len(shape) == 4:
            return Model(inputs=model.input, outputs=layer.output)
    return None

'''
#modello 1
vgg19 = VGG19(weights='imagenet')
vgg19_feature = estraiUltimoLayer(vgg19)
vgg19_alto = estraiUltimoMultidimensionale(vgg19)
print(vgg19_feature.output.shape)
print(vgg19_alto.output.shape)

#modello 2
inceptionResNetV2 = InceptionResNetV2(weights='imagenet')
inceptionResNetV2_feature = estraiUltimoLayer(inceptionResNetV2)
inceptionResNetV2_alto = estraiUltimoMultidimensionale(inceptionResNetV2)
print(inceptionResNetV2_feature.output.shape)
print(inceptionResNetV2_alto.output.shape)

#modello 3
resNet50 = ResNet50(weights='imagenet')
resNet50_feature = estraiUltimoLayer(resNet50)
resNet50_alto = estraiUltimoMultidimensionale(resNet50)
print(resNet50_feature.output.shape)
print(resNet50_alto.output.shape)

#modello 4
efficientNetV2M = EfficientNetV2M(weights='imagenet')
efficientNetV2M_feature = estraiUltimoLayer(efficientNetV2M)
efficientNetV2M_alto = estraiUltimoMultidimensionale(efficientNetV2M)
print(efficientNetV2M_alto.output.shape)
print(efficientNetV2M_feature.output.shape)

#modello 5
convNeXtBase = ConvNeXtBase(weights='imagenet')
convNeXtBase_feature = estraiUltimoLayer(convNeXtBase)
convNeXtBase_alto = estraiUltimoMultidimensionale(convNeXtBase)
print(convNeXtBase_feature.output.shape)
print(convNeXtBase_alto.output.shape)
'''