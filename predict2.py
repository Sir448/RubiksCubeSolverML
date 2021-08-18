import numpy as np
from tensorflow import keras
import h5py
from copy import deepcopy
import sys
import os
import datetime

start = datetime.datetime.now()

modelNum = 0
if len(sys.argv) > 1:
    try:
        modelNum = int(sys.argv[1])
    except:
        modelNum = len(os.listdir("./models"))
else:
    modelNum = len(os.listdir("./models"))

print(f"Loading from model{modelNum}.h5")
model = keras.models.load_model(f"./models/model{modelNum}.h5")

faces = [
    [0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1],
    [2,2,2,2,2,2,2,2,2],
    [3,3,3,3,3,3,3,3,3],
    [4,4,4,4,4,4,4,4,4],
    [5,5,5,5,5,5,5,5,5]
    ]

def oneHotCubeState(cubeFaces):
    pieces = np.arange(6).reshape((-1,1,1,1))
    cubeFaces = np.array(cubeFaces)
    return (cubeFaces == pieces).transpose((1,0,2,3)).astype(int)

def facesToId(faces):
    faces = np.array(faces).reshape(54,)
    output = 0
    for i in faces:
        output = output*6+int(i)
    return 

def idToFaces(id):
    faces = np.array([],dtype='i1')
    for _ in range(54):    
        faces = np.insert(faces,0,id%6)
        id//=6
    return faces.reshape(6,9)

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

#region

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

scrambleAlgStr = "U B' R L F D"
scrambleAlg = scrambleAlgStr.split()


for move in scrambleAlg:
    if move == "F":
        F()
    elif move == "R":
        R()
    elif move == "U":
        U()
    elif move == "L":
        L()
    elif move == "B":
        B()
    elif move == "D":
        D()
    elif move == "F'":
        F(True)
    elif move == "R'":
        R(True)
    elif move == "U'":
        U(True)
    elif move == "L'":
        L(True)
    elif move == "B'":
        B(True)
    elif move == "D'":
        D(True)


moveNotation = ["F","R","U","L","B","D","F'","R'","U'","L'","B'","D'"]

moveFunctions = [F,R,U,L,B,D]

# L = moves from solve
# Put L in array 1
# Iterate through each possible move for each state in array 1 and get L of the next state
# If L is less than previous L, store state, moves from original state, and L in array 2
# Move array 2 to array 1 (Clear array 1, copy array 2 to array 1, delete array 2)
# Repeat till solve or array 2 is empty
def solve():
    global faces
    states1 = []
    states2 = []
    movesAway1 = []
    movesAway2 = []
    moves1 = []
    moves2 = []
    states1.append(deepcopy(faces))
    movesAway1.append(preprocesser(faces))
    moves1.append("")
    for i in range(50):
        startSplit = datetime.datetime.now()
        for state,movesAway,moves in zip(states1,movesAway1,moves1):
            faces = deepcopy(state)
            
            for func,move in zip(moveFunctions,moveNotation[:6]):
                func()
                if checkSolve():
                    print(f"Loaded from model{modelNum}.h5")
                    print("Scramble Alg:",scrambleAlgStr)
                    print("Length of Scramble:",len(scrambleAlg))
                    return moves+" "+move
                L = preprocesser(faces)
                if L <= movesAway:
                    states2.append(deepcopy(faces))
                    movesAway2.append(L)
                    moves2.append(moves+" "+move)
                func(True)
                
            for func,move in zip(moveFunctions,moveNotation[6:]):
                func(True)
                if checkSolve():
                    print(f"Loaded from model{modelNum}.h5")
                    print("Scramble Alg:",scrambleAlgStr)
                    print("Length of Scramble:",len(scrambleAlg))
                    return moves+" "+move
                L = preprocesser(faces)
                if L <= movesAway:
                    states2.append(deepcopy(faces))
                    movesAway2.append(L)
                    moves2.append(moves+" "+move)
                func()
        if len(states2) == 0:
            return "No Solution Found - No lower state"
        else:
            finishSplit = datetime.datetime.now()
            print(f"{len(states2)} in generation number {i+1} | {finishSplit-startSplit}")
        states1 = deepcopy(states2)
        movesAway1 = deepcopy(movesAway2)
        moves1 = deepcopy(moves2)
        states2 = []
        movesAway2 = []
        moves2 = []
    else:
        return "No Solution Found - Reached 50 moves"
    
    

def nextMove(preprocessor):
    global faces
    predictions = []
    F()
    if checkSolve():
        F(True)
        return moveNotation[0]
    predictions.append(preprocessor(faces))
    F(True)
    R()
    if checkSolve():
        R(True)
        return moveNotation[1]
    predictions.append(preprocessor(faces))
    R(True)
    U()
    if checkSolve():
        U(True)
        return moveNotation[2]
    predictions.append(preprocessor(faces))
    U(True)
    L()
    if checkSolve():
        L(True)
        return moveNotation[3]
    predictions.append(preprocessor(faces))
    L(True)
    B()
    if checkSolve():
        B(True)
        return moveNotation[4]
    predictions.append(preprocessor(faces))
    B(True)
    D()
    if checkSolve():
        D(True)
        return moveNotation[5]
    predictions.append(preprocessor(faces))
    D(True)

    F(True)
    if checkSolve():
        F()
        return moveNotation[6]
    predictions.append(preprocessor(faces))
    F()
    R(True)
    if checkSolve():
        R()
        return moveNotation[7]
    predictions.append(preprocessor(faces))
    R()
    U(True)
    if checkSolve():
        U()
        return moveNotation[8]
    predictions.append(preprocessor(faces))
    U()
    L(True)
    if checkSolve():
        L()
        return moveNotation[9]
    predictions.append(preprocessor(faces))
    L()
    B(True)
    if checkSolve():
        B()
        return moveNotation[10]
    predictions.append(preprocessor(faces))
    B()
    D(True)
    if checkSolve():
        D()
        return moveNotation[11]
    predictions.append(preprocessor(faces))
    D()

    return (moveNotation[np.argmin(np.array(predictions))])




def preprocesser(sides):
    return model.predict(oneHotCubeState(faces))[0][0] # Sigmoid
    # return np.argmax(model.predict(oneHotCubeState(faces))[0]) # Onehot

# print(nextMove())

print(solve())

finish=datetime.datetime.now()
print(f"Started at {start}")
print(f"Finished at {finish}")
print(f"Time taken: {finish-start}")