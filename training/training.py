#!/usr/bin/env python3

# Parameters reading
import argparse
from pathlib import Path

import nilearn

from nilearn.image import resample_to_img

import pylab as plt

import numpy as np
import nibabel as nb
import os
import glob
import random
import pandas as pd
import re

import seaborn as sns #added
sns.set(style="darkgrid") #added:1

from nilearn.image import mean_img #added
from nilearn.plotting import plot_anat #added






'''' Get index positions of value in dataframe '''
def getIndexes(dfObj, value):
    listOfPos = list()
    result = dfObj.isin([value]) # Get bool dataframe with True at positions where the given value exists
    seriesObj = result.any() # Get list of columns that contains the value
    columnNames = list(seriesObj[seriesObj == True].index)
    for col in columnNames: # Iterate over list of columns and fetch the rows indexes where value exists
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
# Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

'''DataFrame loading'''
for csv_file in glob.glob('/config/CSV/' + '*.csv'):
    result=pd.read_csv('DataFrame.csv')

'''IMG loading'''
listOfElems=[]  #array that need to obtain the effective subjects with the sequence that we want to analyse 
Dim = []  #array that need to contain the dimension of each image in order to check the most frequent or the minimum one
Data = [] #array that contains information of the image after the extraction of data from the nibabel format
IMG=[] #array that contains the "file.nii.gz" information thanks to the nibabel module 
for Path in result["Path"]: #here starts the iteration on all the paths written in the column "Path" in result dataframe
    #print(Path)
    if os.path.isfile(f"/images/{Path}/{SEQUENCE_1}.nii"): #I need just the Paths that contain a certain sequence

        IMG_reg = nb.load(f"{datasetDir}/{Path}/{SEQUENCE_1}.nii") #nobabel module that allow the loading of nifty file
        DATA= IMG_reg.get_data() #get_data takes the information about scale of gray
        a = [DATA.shape]
        Dim.append(a) #allows the array construction
        Data.append(DATA)  #allows the array construction
        IMG.append(IMG_reg)  #allows the array construction
        

        IMG_roi = nb.load(f"{datasetDir}/{Path}/ROI/{MaskPath}.nii") #that's the file of the mask that limits information only on a certain region of the tumor
        ROI_DATA=IMG_roi.get_data()
        b=[ROI_DATA.shape]
        Dim.append(b)
        Data.append(ROI_DATA)
        IMG.append(IMG_roi)
        
        path=[f"{Path}"]
        listOfElems.append(path) #allows the array construction
        
    else: #anyway there are some patient with the sequence with different name but with the same information, so I can include them with the "else"
        if os.path.isfile(f"{datasetDir}/{Path}/{SEQUENCE_2}.nii"):
            IMG_reg = nb.load(f"{datasetDir}/{Path}/{SEQUENCE_2}.nii")
            DATA= IMG_reg.get_data()
            a = [DATA.shape]
            Dim.append(a)
            Data.append(DATA)
            IMG.append(IMG_reg)
            
            IMG_roi = nb.load(f"{datasetDir}/{Path}/ROI/{MaskPath}.nii")
            ROI_DATA=IMG_roi.get_data()
            b=[ROI_DATA.shape]
            Dim.append(b)
            Data.append(ROI_DATA)
            IMG.append(IMG_roi)
            
            path=[f"{Path}"]
            listOfElems.append(path)
            
            
'''Labels Array Construction'''

#the following iteration allows the extraction of indexes corresponding to the paths selected by the preavious iterations
T1_Subject=[]
for i in range(len(listOfElems)):
    Pos=getIndexes(result, listOfElems[i][0])
    print(Pos[0][0])
    T1_Subject.append(Pos[0][0])  #the T1_Subject array contains these information

#the following iteration takes the corresponding Survival information on the result Dataframe built before    
T1_Subject_array=np.asarray(T1_Subject)

T1_Labels=[]
for i in range(len(T1_Subject_array)):
    lab = result.loc[T1_Subject_array[i],"Survival"]
    T1_Labels.append(int(lab)) #creates the list that contains these information

T1_Labels=np.asarray(T1_Labels) #that's the relative array
#T1_Labels

#the following iteration allows the construction of the final labels array with doubled information of each element of T1_Labels array
Label_Def=[]
for x in range(len(T1_Labels)):
    label_Def=[[T1_Labels[x]]*2]
    Label_Def.append(label_Def)
    
Label_Def=np.asarray(Label_Def)
Label_Def=np.ravel(Label_Def)
Label_Def #that's the final label array that can be used for the training



'''Working on dimensions'''

#in order to work on dimensions it's necessary the array construction (as predicted in previous blocks)
Dim=np.asarray(Dim)   
Data=np.asarray(Data)
IMG=np.asarray(IMG)

#in order to know the minimum dimensions
Min_value=np.amin(Dim, axis=1)
Min_value
Min=np.amin(Min_value, axis=0)
Min_value.shape[0]

not_in_index = [k for k in range(Min_value.shape[0]) if not np.all(Min_value[k] == (dim1, dim2, dim3))] #that's the
#construction of an array that contains the position not corresponding to the dimension researched [(192,256,144)
#in this case]

pos_1=np.where(Min_value[:,0]==dim1) #position with first dimension equal to 192
pos_2=np.where(Min_value[:,1]==dim2) #position with first dimension equal to 256
pos_3=np.where(Min_value[:,2]==dim3) #position with first dimension equal to 144
eq=np.intersect1d(pos_1,pos_2)  #that command in order to find the intersection between the pos_1 and pos_2
index_IMG=np.intersect1d(eq,pos_3) #intersection that gives the complememntary information of not_in_index

#Here is given a random position that corresponds to the dimension request
def_index=random.choice(index_IMG)
print(def_index)



'''RESAMPLE BLOCK'''

#The resample function is executed only on images without the dimension request, respect to a random image with dimension (192,256,144)
for i in not_in_index:
    Res=nilearn.image.resample_to_img(IMG[i], IMG[def_index])
    IMG[i]=Res
    Data[i]=IMG[i].get_data()
    
    
    
'''Reshaping of Input Matrix'''

Input_matrix=np.empty((len(Data),dim1,dim2,dim3)) #in order to generate an empty array with a fixed shape

for i in not_in_index:

    Input_matrix[i,:,:,:]=np.array(Data[i])




for i in index_IMG:

    Input_matrix[i,:,:,:]=np.array(Data[i])
    
    
#in order to check the correct construction    
Input_matrix.shape

#import pickle
#pickle.dump( Input_matrix, open( "Input_matrix.pickle", "wb" ) )


'''Modules needed for TRAINING'''

import tensorflow as tf
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam, SGD

from keras import backend as K


## Create list of indices and shuffle them
N = Input_matrix.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)

#  Cut the dataset at 80% to create the training and test set
N_80p = int(p * N)
indices_train = indices[:N_80p]
indices_test = indices[N_80p:]

# Split the data into training and test sets
X_train = Input_matrix[indices_train, ...]
X_test = Input_matrix[indices_test, ...]



labels=Label_Def

#Outcome variable block added
y_train = labels[indices_train] == 0
y_test = labels[indices_test] == 1

from keras.utils import to_categorical #Convert a class vector (integer) into binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


'''Parameters setting'''

# Get shape of input data
data_shape = tuple(X_train.shape[1:])

#model block added
K.clear_session()
model = Sequential()

model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=data_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(filters * 2, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(filters * 4, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

# optimizer
#learning_rate = 1e-5
adam = Adam(lr=learning_rate)
sgd = SGD(lr=learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=adam, # swap out for sgd 
              metrics=['accuracy'])

model.summary()



'''Model fitting'''

%time fit = model.fit(X_train, y_train, epochs=nEpochs, batch_size=batch_size)


'''PLOT'''

fig = plt.figure(figsize=(10, 4))
epoch = np.arange(nEpochs) + 1
fontsize = 16
plt.plot(epoch, fit.history['accuracy'], marker="o", linewidth=2,
         color="steelblue", label="accuracy")
plt.plot(epoch, fit.history['loss'], marker="o", linewidth=2,
         color="orange", label="loss")
plt.xlabel('epoch', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(frameon=False, fontsize=16);

parser = argparse.ArgumentParser()


# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--learning-rate', type=str, help='Learning rate') # Paths should be passed in, not hardcoded
parser.add_argument('--batch-size', type=int, default=16, help='Parameter 1.')
parser.add_argument('--dim1', type=int, default=192, help='Dimension1')
parser.add_argument('--dim2', type=int, default=256, help='Dimension2')
parser.add_argument('--dim3', type=int, default=144, help='Dimension3')
parser.add_argument('--SEQUENCE_1', type=str)
parser.add_argument('--SEQUENCE_2', type=str)
parser.add_argument('--MaskPath', type=str)
parser.add_argument('--p', type=float, default=0.8, help='DataPercentage')
parser.add_argument('--kernel_size', nargs='+', type=int, help='KernelDim')
parser.add_argument('--n_classes', type=int, default=2, help='Classes')
parser.add_argument('--filters', type=int, default=16, help='Filters')
parser.add_argument('--nEpochs', type=int, default=100, help='Epochs')

args = parser.parse_args()
kernel_size_tuple = tuple(args.kernel_size)

print(f"I'm running training with a learning rate of: {args.learning_rate}")
print(f"Batch size if {args.batch_size}")
print(f"With dim1 equal to {args.dim1}")
print(f"With dim2 equal to {args.dim2}")
print(f"With dim3 equal to {args.dim3}")
print(f"I'm running training with the following sequence1 {args.SEQUENCE_1}")
print(f"I'm running training with the following sequence2 {args.SEQUENCE_2}")
print(f"I'm running training on the following mask {args.MaskPath}")
print(f"I'm considering this percentage of data for the training {args.p}")
print(f"The number of output classes is {args.n_classes}")
print(f"I'm considering the following number of filters {args.filters}")
print(f"I'm running the following number of epochs {args.nEpochs}")









