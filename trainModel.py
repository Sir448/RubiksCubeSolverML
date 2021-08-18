import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import numpy as np
import h5py
import datetime
from math import ceil
import os
import sys



batchSize = 5000

import sys

modelNum = 0
if len(sys.argv) > 1:
    try:
        modelNum = int(sys.argv[1])
    except:
        modelNum = len(os.listdir("./models"))+1
else:
    modelNum = len(os.listdir("./models"))+1
    
    
filename = f'model{modelNum}.h5'
print(f"Saving to {filename}")

start = datetime.datetime.now()


#region

# cube3d = layers.Input(shape=(6, 6, 9))

# x = cube3d

# for _ in range(4):
#     x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', data_format='channels_first')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64,'relu')(x)
# x = layers.Dense(1,'sigmoid')(x)




# x = cube3d

# for _ in range(4):
#     x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', data_format='channels_first')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(54,'relu')(x)
# x = layers.Dense(27,'softmax')(x)




# x = cube3d
# x = layers.Dense(6*9,'tanh')(x)
# x = layers.Dense(6*9,'relu')(x)
# x = layers.Dropout(0.3)(x)
# x = layers.Dense(6*9,'relu')(x)
# x = layers.Dense(27,'softmax')(x)



# cube3d = layers.Input(shape=(6,6,9))

# x = cube3d

# for _ in range(4):
    # x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', data_format='channels_first')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(54,'tanh')(x)
# x = layers.Dense(54,'relu')(x)
# x = layers.Dropout(0.3)(x)
# x = layers.Dense(54,'relu')(x)
# x = layers.Dense(27,'softmax')(x)

#endregion


cube3d = layers.Input(shape=(6,6,9))

x = cube3d

for _ in range(4):
    x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', data_format='channels_first')(x)
x = layers.Flatten()(x)
x = layers.Dense(54,'tanh')(x)
x = layers.Dense(54,'relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(54,'relu')(x)
# x = layers.Dense(27,'softmax')(x)
x = layers.Dense(1,'sigmoid')(x)

# conv2d vs dense
# tanh vs relu
# softmax vs sigmoid


model = keras.Model(inputs=cube3d,outputs=x)

model.compile(optimizer=optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy") #Index
# model.compile(optimizer=optimizers.Adam(5e-4), loss="categorical_crossentropy") #One hot
# model.compile(optimizer=optimizers.Adam(5e-4), loss="mean_squared_error") #Index

model.summary()

f = h5py.File('solveDataGood.hdf5','r')
# f = h5py.File('solveData.hdf5','r')


numberOfRounds = ceil(f['moves'].shape[0]/batchSize)
# faces = f['faces'][...]
# moves = f['moves'][...]
# faces = f['faces'][:11206]
# moves = f['moves'][:11206]
# fitModel = model.fit(faces,moves,epochs=5)

# model.save(filename)

for i in range(numberOfRounds):
    
    faces = f['faces'][i*batchSize:(i+1)*batchSize]
    # faces = faces.reshape(faces.shape[0],324)
    # faces = np.argmax(f['faces'][i*batchSize:(i+1)*batchSize].transpose(0,3,2,1),axis=3)
    # faces = faces.reshape(faces.shape[0],54)
    
    moves = f['moves'][i*batchSize:(i+1)*batchSize]/26
    # moves = f['moves'][i*batchSize:(i+1)*batchSize]
    # oneHotMoves = np.zeros((moves.shape[0],27))
    # oneHotMoves[np.arange(moves.size),moves] = 1
    fitModel = model.fit(faces,moves)
    # fitModel = model.train_on_batch(faces,moves)
    del faces
    del moves
    f.flush()
    # del oneHotMoves
    
    model.save("./models/"+filename)
    print(f'Finished training round #{i+1} of {numberOfRounds}')
    
# faces = f['faces'][:11206]
# moves = f['moves'][:11206]/26
# # oneHotMoves = np.zeros((moves.shape[0],27))
# # oneHotMoves[np.arange(moves.size),moves] = 1
# fitModel = model.fit(faces,moves,epochs=5,batch_size=50)


# faces = f['faces'][:144]
# moves = f['moves'][:144]


# fitModel = model.fit(faces,moves,epochs=100,batch_size=50)


model.save("./models/"+filename)


f.close()

finish = datetime.datetime.now()

print("Started at",start)
print("Finished at",finish)
print("Time taken", finish-start)
print(f"Saved to {filename}")
