import numpy as np
from copy import deepcopy
from time import time, ctime

scrambleAlgStr = "U F"
scrambleAlg = scrambleAlgStr.split()



#region backup solve state
faces = [
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

scramble = deepcopy(faces)


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

# 14,8,8
# 6,6,9

pieces = np.arange(6).reshape((-1,1,1,1))
faces = np.array(faces)
oneHotFaces = (faces == pieces).transpose((1,0,2,3)).astype(int)