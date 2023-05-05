import random


class Vehicle(object):
    def __init__(self, idd):
        self.ID = idd


class Base(object):
    def __init__(self, st, cap):
        self.name = st
        self.maxVehicles = cap
        self.residualCapacity = cap
        self.hostedVehicles = []


class Matching(object):
    def __init__(self):
        self.criterionScore = 100000
        self.baseIndex = -1
        self.vehicleIndex = -1


class Locator:
    def __init__(self):
        self.vehicles = []
        self.bases = []
        self.distanceMatrix = []
        self.totalDistance = 0

    def BuildInput(self):
        random.seed(1)

        for i in range (0, 10):
            v = Vehicle(len(self.vehicles))
            self.vehicles.append(v)

        b = Base('A', 4)
        b.ID = len(self.bases)
        self.bases.append(b)
        b = Base('B', 6)
        b.ID = len(self.bases)
        self.bases.append(b)
        b = Base('C', 5)
        b.ID = len(self.bases)
        self.bases.append(b)
        b = Base('D', 8)
        b.ID = len(self.bases)
        self.bases.append(b)
        b = Base('E', 4)
        self.bases.append(b)
        b = Base('F', 5)
        self.bases.append(b)

        self.BuildDistanceMatrix()

    def BuildDistanceMatrix(self):
        rows = len(self.bases)
        cols = len(self.vehicles)
        self.distanceMatrix = [[0.0 for x in range(cols)] for y in range(rows)]

        self.distanceMatrix[0][0] = 72
        self.distanceMatrix[0][1] = 36
        self.distanceMatrix[0][2] = 89
        self.distanceMatrix[0][3] = 46
        self.distanceMatrix[0][4] = 130
        self.distanceMatrix[0][5] = 56
        self.distanceMatrix[0][6] = 61
        self.distanceMatrix[0][7] = 35
        self.distanceMatrix[0][8] = 34
        self.distanceMatrix[0][9] = 56

        self.distanceMatrix[1][0] = 31
        self.distanceMatrix[1][1] = 140
        self.distanceMatrix[1][2] = 79
        self.distanceMatrix[1][3] = 63
        self.distanceMatrix[1][4] = 78
        self.distanceMatrix[1][5] = 31
        self.distanceMatrix[1][6] = 25
        self.distanceMatrix[1][7] = 92
        self.distanceMatrix[1][8] = 128
        self.distanceMatrix[1][9] = 29

        self.distanceMatrix[2][0] = 56
        self.distanceMatrix[2][1] = 53
        self.distanceMatrix[2][2] = 45
        self.distanceMatrix[2][3] = 88
        self.distanceMatrix[2][4] = 25
        self.distanceMatrix[2][5] = 44
        self.distanceMatrix[2][6] = 33
        self.distanceMatrix[2][7] = 37
        self.distanceMatrix[2][8] = 74
        self.distanceMatrix[2][9] = 46

        self.distanceMatrix[3][0] = 78
        self.distanceMatrix[3][1] = 68
        self.distanceMatrix[3][2] = 28
        self.distanceMatrix[3][3] = 42
        self.distanceMatrix[3][4] = 141
        self.distanceMatrix[3][5] = 72
        self.distanceMatrix[3][6] = 45
        self.distanceMatrix[3][7] = 78
        self.distanceMatrix[3][8] = 28
        self.distanceMatrix[3][9] = 89

        self.distanceMatrix[4][0] = 160
        self.distanceMatrix[4][1] = 86
        self.distanceMatrix[4][2] = 74
        self.distanceMatrix[4][3] = 71
        self.distanceMatrix[4][4] = 137
        self.distanceMatrix[4][5] = 83
        self.distanceMatrix[4][6] = 61
        self.distanceMatrix[4][7] = 89
        self.distanceMatrix[4][8] = 179
        self.distanceMatrix[4][9] = 91

        self.distanceMatrix[5][0] = 95
        self.distanceMatrix[5][1] = 41
        self.distanceMatrix[5][2] = 34
        self.distanceMatrix[5][3] = 122
        self.distanceMatrix[5][4] = 39
        self.distanceMatrix[5][5] = 52
        self.distanceMatrix[5][6] = 54
        self.distanceMatrix[5][7] = 22
        self.distanceMatrix[5][8] = 45
        self.distanceMatrix[5][9] = 23

    def Solve(self):
        vehiclesToBeAssignedToBases = self.vehicles.copy()
        usedBases = []

        for i in range(0, len(self.vehicles)):
            bestMatching = self.FindBestMatching(vehiclesToBeAssignedToBases, usedBases)
            if bestMatching.baseIndex != -1:
                self.ApplyBestMatching(bestMatching, vehiclesToBeAssignedToBases, usedBases)
            else:
                print('Infeasibility')
                # infeasibility issue

    def FindBestMatching(self, vehiclesToBeAssignedToBases, usedBases):
        match = Matching()
        for i in range (0, len(self.bases)):
            b = self.bases[i]
            if self.ViolatesTheMaximumBasesToBeUsed(b, usedBases) == True:
                continue
            if b.residualCapacity == 0:
                continue

            for j in range (0, len(vehiclesToBeAssignedToBases)):
                v = vehiclesToBeAssignedToBases[j]

                score = self.distanceMatrix[i][v.ID]

                if score < match.criterionScore:
                    match.criterionScore = score
                    match.baseIndex = i
                    match.vehicleIndex = j

        return match

    def ViolatesTheMaximumBasesToBeUsed(self, b, usedBases):
        if usedBases.count(b) == 0 and len(usedBases) == 3:
            return True
        return False

    def ApplyBestMatching(self, match, vehiclesToBeAssignedToBases, usedBases):
        b = self.bases[match.baseIndex]
        v = vehiclesToBeAssignedToBases[match.vehicleIndex]
        b.hostedVehicles.append(v)
        b.residualCapacity = b.residualCapacity - 1
        if usedBases.count(b) == 0:
            usedBases.append(b)
        vehiclesToBeAssignedToBases.remove(v)
        self.totalDistance = self.totalDistance +  match.criterionScore








