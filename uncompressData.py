import json
import numpy as np
import h5py

g = h5py.File('solveDataRepeats3.hdf5','a')

def save_data(names,data,dtypes):
    global g
    # Load file
    data = [np.array(d, dtype='i1') for d in data]
    for name,d,dtype in zip(names,data,dtypes):
        # Load or create dataset
        if name not in g.keys():
            dataset = g.create_dataset(name,shape=(0,)+d.shape[1:],maxshape=(None,)+d.shape[1:],chunks=True,dtype=dtype)
        else:
            dataset = g[name]
        dataset.resize((dataset.shape[0]+len(d)),axis=0)
        dataset[-len(d):] = d
        


pieces = np.arange(6).reshape((-1,1,1,1))

def oneHotCubeState(cubeFaces):
    cubeFaces = np.array(cubeFaces, dtype='i1')
    return (cubeFaces == pieces).transpose((1,0,2,3)).astype('i1')


def idToFaces(id):
    faces = np.array([],dtype='i1')
    for _ in range(54):    
        faces = np.insert(faces,0,id%6)
        id//=6
    return faces.reshape(6,9)

def idToOneHot(id):
    return oneHotCubeState(idToFaces(id))

with open("solveDataRepeatsCompressed.json",'r') as f:
    data = json.load(f)
    
states = np.array([],dtype='i1').reshape(0,6,6,9)
moves = np.array([],dtype='i1')

# stateNumber = 0
rounds = 0

for key in data.keys():
    oneHot = idToOneHot(int(key))
    moveCount = data[key][0]
    for _ in range(data[key][1]):
        states = np.append(states,oneHot,axis=0)
        moves = np.append(moves,moveCount)
        if len(moves) >= 5000:
            save_data(['moves','faces'],[moves,states],['i1','i1'])
            states = np.array([],dtype='i1').reshape(0,6,6,9)
            moves = np.array([],dtype='i1')
            rounds+=1
            print(f"Saved {rounds} rounds")
    # stateNumber += 1
    # print(f"Finished {stateNumber} out of {len(data)}")
    # if stateNumber % 100 == 0:

if len(moves) > 0:
    save_data(['moves','faces'],[moves,states],['i1','i1'])

# print("Shuffling")
# # seed = np.random.randint(1000000)
# seed = 1000000
# np.random.seed(seed)
# np.random.shuffle(f['moves'])
# np.random.seed(seed)
# np.random.shuffle(f['faces'])

# print("Saving")

g.close()
