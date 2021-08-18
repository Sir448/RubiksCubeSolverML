from copy import deepcopy
from time import time, ctime
import numpy as np
from random import randint
import h5py
from math import ceil

dataLength = int(input("How much data do you want? "))

start = update = time()


#region backup solve state
faces = [
    [0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [2,2,2,2,2,2,2,2,2],
    [3,3,3,3,3,3,3,3,3],
    [4,4,4,4,4,4,4,4,4],
    [5,5,5,5,5,5,5,5,5]
    ]

solvedFaces = [
    [0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [2,2,2,2,2,2,2,2,2],
    [3,3,3,3,3,3,3,3,3],
    [4,4,4,4,4,4,4,4,4],
    [5,5,5,5,5,5,5,5,5]
    ]

#endregion

#region colour/side/moves legend

# yellow = 0
# green = 1
# orange = 2
# blue = 3
# red = 4
# white = 5

# bottom = 0
# front = 1
# left = 2
# back = 3
# right = 4
# top = 5

# F = 0
# R = 1
# U = 2
# L = 3
# B = 4
# D = 5
# E = 6
# M = 7
# S = 8
# F' = 9
# R' = 10
# U' = 11
# L' = 12
# B' = 13
# D' = 14
# E' = 15
# M' = 16
# S' = 17
# F2 = 18
# R2 = 19
# U2 = 20
# L2 = 21
# B2 = 22
# D2 = 23
# E2 = 24
# M2 = 25
# S2 = 26

#endregion

#region moves

def F(prime = False):
    global faces
    if not prime:
        faces[1] = faces[1][-3:-1] + faces[1][:-3] + [faces[1][8]]
        cache = [faces[0][0],faces[0][1],faces[0][2]]
        faces[0][0] = faces[4][6]
        faces[0][1] = faces[4][7]
        faces[0][2] = faces[4][0]
        faces[4][6] = faces[5][4]
        faces[4][7] = faces[5][5]
        faces[4][0] = faces[5][6]
        faces[5][4] = faces[2][2]
        faces[5][5] = faces[2][3]
        faces[5][6] = faces[2][4]
        faces[2][2] = cache[0]
        faces[2][3] = cache[1]
        faces[2][4] = cache[2]
    else:
        faces[1] = faces[1][2:-1] + faces[1][:2] + [faces[1][8]]
        cache = [faces[0][0],faces[0][1],faces[0][2]]
        faces[0][0] = faces[2][2]
        faces[0][1] = faces[2][3]
        faces[0][2] = faces[2][4]
        faces[2][2] = faces[5][4]
        faces[2][3] = faces[5][5]
        faces[2][4] = faces[5][6]
        faces[5][4] = faces[4][6]
        faces[5][5] = faces[4][7]
        faces[5][6] = faces[4][0]
        faces[4][6] = cache[0]
        faces[4][7] = cache[1]
        faces[4][0] = cache[2]

def R(prime = False):
    global faces
    if not prime:
        faces[4] = faces[4][-3:-1] + faces[4][:-3] + [faces[4][8]]
        cache = [faces[0][2],faces[0][3],faces[0][4]]
        faces[0][2] = faces[3][6]
        faces[0][3] = faces[3][7]
        faces[0][4] = faces[3][0]
        faces[3][6] = faces[5][2]
        faces[3][7] = faces[5][3]
        faces[3][0] = faces[5][4]
        faces[5][2] = faces[1][2]
        faces[5][3] = faces[1][3]
        faces[5][4] = faces[1][4]
        faces[1][2] = cache[0]
        faces[1][3] = cache[1]
        faces[1][4] = cache[2]
    else:
        faces[4] = faces[4][2:-1] + faces[4][:2] + [faces[4][8]]
        cache = [faces[0][2],faces[0][3],faces[0][4]]
        faces[0][2] = faces[1][2]
        faces[0][3] = faces[1][3]
        faces[0][4] = faces[1][4]
        faces[1][2] = faces[5][2]
        faces[1][3] = faces[5][3]
        faces[1][4] = faces[5][4]
        faces[5][2] = faces[3][6]
        faces[5][3] = faces[3][7]
        faces[5][4] = faces[3][0]
        faces[3][6] = cache[0]
        faces[3][7] = cache[1]
        faces[3][0] = cache[2]
    
def U(prime = False):
    global faces
    if not prime:
        faces[5] = faces[5][-3:-1] + faces[5][:-3] + [faces[5][8]]
        cache = [faces[4][0],faces[4][1],faces[4][2]]
        for i in range(3):
            for j in range(3):
                faces[4-i][j] = faces[4-i-1][j]
        for i in range(3):
            faces[1][i] = cache[i]
    else:
        faces[5] = faces[5][2:-1] + faces[5][:2] + [faces[5][8]]
        cache = [faces[1][0],faces[1][1],faces[1][2]]
        for i in range(1,4):
            for j in range(3):
                faces[i][j] = faces[i+1][j]
        for i in range(3):
            faces[4][i] = cache[i]

def L(prime = False):
    global faces
    if not prime:
        faces[2] = faces[2][-3:-1] + faces[2][:-3] + [faces[2][8]]
        cache = [faces[0][6],faces[0][7],faces[0][0]]
        faces[0][6] = faces[1][6]
        faces[0][7] = faces[1][7]
        faces[0][0] = faces[1][0]
        faces[1][6] = faces[5][6]
        faces[1][7] = faces[5][7]
        faces[1][0] = faces[5][0]
        faces[5][6] = faces[3][2]
        faces[5][7] = faces[3][3]
        faces[5][0] = faces[3][4]
        faces[3][2] = cache[0]
        faces[3][3] = cache[1]
        faces[3][4] = cache[2]
    else:
        faces[2] = faces[2][2:-1] + faces[2][:2] + [faces[2][8]]
        cache = [faces[0][6],faces[0][7],faces[0][0]]
        faces[0][6] = faces[3][2]
        faces[0][7] = faces[3][3]
        faces[0][0] = faces[3][4]
        faces[3][2] = faces[5][6]
        faces[3][3] = faces[5][7]
        faces[3][4] = faces[5][0]
        faces[5][6] = faces[1][6]
        faces[5][7] = faces[1][7]
        faces[5][0] = faces[1][0]
        faces[1][6] = cache[0]
        faces[1][7] = cache[1]
        faces[1][0] = cache[2]
    
def B(prime = False):
    global faces
    if not prime:
        faces[3] = faces[3][-3:-1] + faces[3][:-3] + [faces[3][8]]
        cache = [faces[0][4],faces[0][5],faces[0][6]]
        faces[0][4] = faces[2][6]
        faces[0][5] = faces[2][7]
        faces[0][6] = faces[2][0]
        faces[2][6] = faces[5][0]
        faces[2][7] = faces[5][1]
        faces[2][0] = faces[5][2]
        faces[5][0] = faces[4][2]
        faces[5][1] = faces[4][3]
        faces[5][2] = faces[4][4]
        faces[4][2] = cache[0]
        faces[4][3] = cache[1]
        faces[4][4] = cache[2]
    else:
        faces[3] = faces[3][2:-1] + faces[3][:2] + [faces[3][8]]
        cache = [faces[0][4],faces[0][5],faces[0][6]]
        faces[0][4] = faces[4][2]
        faces[0][5] = faces[4][3]
        faces[0][6] = faces[4][4]
        faces[4][2] = faces[5][0]
        faces[4][3] = faces[5][1]
        faces[4][4] = faces[5][2]
        faces[5][0] = faces[2][6]
        faces[5][1] = faces[2][7]
        faces[5][2] = faces[2][0]
        faces[2][6] = cache[0]
        faces[2][7] = cache[1]
        faces[2][0] = cache[2]
    
def D(prime = False):
    global faces
    if not prime:
        faces[0] = faces[0][-3:-1] + faces[0][:-3] + [faces[0][8]]
        cache = [faces[1][4],faces[1][5],faces[1][6]]
        for i in range(1,4):
            for j in range(4,7):
                faces[i][j] = faces[i+1][j]
        for i in range(3):
            faces[4][i+4] = cache[i]
    else:
        faces[0] = faces[0][2:-1] + faces[0][:2] + [faces[0][8]]
        cache = [faces[4][4],faces[4][5],faces[4][6]]
        for i in range(3):
            for j in range(4,7):
                faces[4-i][j] = faces[4-i-1][j]
        for i in range(3):
            faces[1][i+4] = cache[i]

#endregion


pieces = np.arange(6).reshape((-1,1,1,1))


def oneHotCubeState(cubeFaces):
    pieces = np.arange(6).reshape((-1,1,1,1))
    cubeFaces = np.array(cubeFaces, dtype='i1')
    return (cubeFaces == pieces).transpose((1,0,2,3)).astype('i1')

f = h5py.File("solveDataRepeats.hdf5","a")

def save_data(names,data,dtypes):
    global f
    # Load file
    data = [np.array(d, dtype='i1') for d in data]
    for name,d,dtype in zip(names,data,dtypes):
        # Load or create dataset
        if name not in f.keys():
            dataset = f.create_dataset(name,shape=(0,)+d.shape[1:],maxshape=(None,)+d.shape[1:],chunks=True,dtype=dtype)
        else:
            dataset = f[name]
        dataset.resize((dataset.shape[0]+len(d)),axis=0)
        dataset[-len(d):] = d



def checkSolve():
    global faces
    for face in faces:
        for piece in range(len(face)-1):
            if face[piece] == face[piece+1]:
                continue
            break
        else:
            continue
        break
    else:
        return True
    return False

def facesToId(faces):
    faces = np.array(faces, dtype='i1').reshape(54,)
    output = 0
    for i in faces:
        output = output*6+int(i)
    return output
    
def check(test,array):
    for i,x in enumerate(array):
        if np.array_equal(x,test):
            return i
    return -1

numberOfMoves = np.array([0], dtype='i1')
states = oneHotCubeState(faces)
ids = {facesToId(faces):np.array([0])}
total = 1

i = 0
            
def addState(move):
    global i, states, numberOfMoves, faces, solvedFaces, ids, f, total
    if move == 0:
        F()
    elif move == 1:
        R()
    elif move == 2:
        U()
    elif move == 3:
        L()
    elif move == 4:
        B()
    elif move == 5:
        D()
    elif move == 6:
        F(True)
    elif move == 7:
        R(True)
    elif move == 8:
        U(True)
    elif move == 9:
        L(True)
    elif move == 10:
        B(True)
    elif move == 11:
        D(True)
    i += 1
        
    currentState = oneHotCubeState(faces)
    Id = facesToId(faces)
    
    if Id not in ids.keys():
        numberOfMoves = np.append(numberOfMoves,i)
        states = np.append(states,currentState,axis=0)
        ids[Id] = np.array([total])
        total += 1
    else:
        index = ids[Id]
        savedIndex = index[index<(total-len(numberOfMoves))]
        unsavedIndex = index[index>=(total-len(numberOfMoves))]-total+len(numberOfMoves)
        minMoves = None
        if len(savedIndex) > 0:
            minMoves = min(f['moves'][savedIndex[0]],i)
            f['moves'][savedIndex] = minMoves
        else:
            minMoves = min(numberOfMoves[unsavedIndex[0]],i)
        if len(unsavedIndex) > 0:
            numberOfMoves[unsavedIndex] = minMoves
    
        numberOfMoves = np.append(numberOfMoves,i)
        states = np.append(states,currentState,axis=0)
        ids[Id] = np.append(ids[Id],total)
        total += 1
            
    # if Id not in ids.keys():
    #     numberOfMoves.append(i)
    #     states = np.append(states,currentState,axis=0)
    #     ids[Id] = [total]
    #     total += 1
    # else:
    #     for index in ids[Id]:
    #         if total != len(numberOfMoves) and index >= total - len(numberOfMoves): # There is saved data and index is saved
    #             index += len(numberOfMoves) - total
    #             numberOfMoves[index] = min(i,numberOfMoves[index])
    #         elif total != len(numberOfMoves): # There is saved data but index isn't saved
    #             f['moves'][index] = min(i,f['moves'][index])
    #         else: # Index is in saved set
    #             numberOfMoves[index] = min(i,numberOfMoves[index])
    
    #     numberOfMoves.append(i)
    #     states = np.append(states,currentState,axis=0)
    #     ids[Id].append(total)
    #     total+=1
         
    if i >= 26:
        reset()

def reset():
    global faces, states, total, i, numberOfMoves, ids
    i = 0
    faces = deepcopy(solvedFaces)
    Id = facesToId(faces)
    numberOfMoves = np.append(numberOfMoves,0)
    states = np.append(states,oneHotCubeState(faces),axis=0)
    ids[Id] = np.append(ids[Id],total)
    total += 1
    
    

for j in range(12):
    print(f"Starting round number {j+1} of iterative scrambles")
    for k in range(12):
        for l in range(12):
            for m in range(12):
                addState(j)
                addState(k)
                addState(l)
                addState(m)
                i = 0
                faces = deepcopy(solvedFaces)


save_data(['moves','faces'],[numberOfMoves,states],['i1','i1'])
numberOfMoves = np.array([0], dtype='i1')
states = oneHotCubeState(faces)

rounds = ceil(dataLength/5000)

print("Starting random scrambles")
# saves = 0
while total < dataLength:
    
    addState(randint(0,11))
    if len(numberOfMoves) >= 5001:
        print(f"On round {total//5000} of {rounds}")            
        save_data(['moves','faces'],[numberOfMoves,states],['i1','i1'])
        numberOfMoves = np.array([0], dtype='i1')
        states = oneHotCubeState(faces)
        
save_data(['moves','faces'],[numberOfMoves,states],['i1','i1'])

f.close()