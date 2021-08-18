from random import randint
alg = []
moves = int(input("How many moves in scramble algorithm? "))
while moves > 0:
    r = randint(0,11)
    if r == 0:
        if len(alg) > 0:
            if alg[-1] == "F'":
                continue
        if len(alg) > 1:
            if alg[-1] == "F" and alg[-2] == "F":
                continue
        alg.append("F")
    elif r == 1:
        if len(alg) > 0:
            if alg[-1] == "R'":
                continue
        if len(alg) > 1:
            if alg[-1] == "R" and alg[-2] == "R":
                continue
        alg.append("R")
    elif r == 2:
        if len(alg) > 0:
            if alg[-1] == "U'":
                continue
        if len(alg) > 1:
            if alg[-1] == "U" and alg[-2] == "U":
                continue
        alg.append("U")
    elif r == 3:
        if len(alg) > 0:
            if alg[-1] == "L'":
                continue
        if len(alg) > 1:
            if alg[-1] == "L" and alg[-2] == "L":
                continue
        alg.append("L")
    elif r == 4:
        if len(alg) > 0:
            if alg[-1] == "B'":
                continue
        if len(alg) > 1:
            if alg[-1] == "B" and alg[-2] == "B":
                continue
        alg.append("B")
    elif r == 5:
        if len(alg) > 0:
            if alg[-1] == "D'":
                continue
        if len(alg) > 1:
            if alg[-1] == "D" and alg[-2] == "D":
                continue
        alg.append("D")
    elif r == 6:
        if len(alg) > 0:
            if alg[-1] == "F":
                continue
        if len(alg) > 1:
            if alg[-1] == "F'" and alg[-2] == "F'":
                continue
        alg.append("F'")
    elif r == 7:
        if len(alg) > 0:
            if alg[-1] == "R":
                continue
        if len(alg) > 1:
            if alg[-1] == "R'" and alg[-2] == "R'":
                continue
        alg.append("R'")
    elif r == 8:
        if len(alg) > 0:
            if alg[-1] == "U":
                continue
        if len(alg) > 1:
            if alg[-1] == "U'" and alg[-2] == "U'":
                continue
        alg.append("U'")
    elif r == 9:
        if len(alg) > 0:
            if alg[-1] == "L":
                continue
        if len(alg) > 1:
            if alg[-1] == "L'" and alg[-2] == "L'":
                continue
        alg.append("L'")
    elif r == 10:
        if len(alg) > 0:
            if alg[-1] == "B":
                continue
        if len(alg) > 1:
            if alg[-1] == "B'" and alg[-2] == "B'":
                continue
        alg.append("B'")
    elif r == 11:
        if len(alg) > 0:
            if alg[-1] == "D":
                continue
        if len(alg) > 1:
            if alg[-1] == "D'" and alg[-2] == "D'":
                continue
        alg.append("D'")
    moves -= 1
print(" ".join(alg))
# input()