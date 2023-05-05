import random


class Location(object):
    def __init__(self, str, id):
        self.indexInTheList = id
        self.name = str


class Warehouse(object):
    def __init__(self, str, id):
        self.indexInTheList = id
        self.name = str
        self.costComponent = 0


class QuadraticAssignmentProblem:
    def __init__(self):
        self.distanceMatrix = []
        self.costPerDistanceUnit = []
        self.warehouses = []
        self.locations = []

    def BuildExampleModel(self):
        self.BuildLocations()
        self.BuildDistanceMatrix()
        self.BuildWarehouses()
        self.BuildCostMatrix()

    def BuildRandomModel(self):
        random.seed(1)
        totalLocations = 20 #50 loc --> times: 150 8 4 / 100 locs --> times: - 142 75
        self.BuildRandomLocations(totalLocations)
        self.BuildRandomDistanceMatrix(totalLocations)
        self.BuildRandomWarehouses(totalLocations)
        self.BuildRandomCostMatrix(totalLocations)

    def BuildLocations(self):
        l = Location('A', len(self.locations))
        self.locations.append(l)
        l = Location('B', len(self.locations))
        self.locations.append(l)
        l = Location('C', len(self.locations))
        self.locations.append(l)
        l = Location('D', len(self.locations))
        self.locations.append(l)

    def BuildRandomLocations(self, totalLocations):
        for i in range(0, totalLocations):
            l = Location('L'+str(i), len(self.locations))
            self.locations.append(l)

    def BuildWarehouses(self):
        w = Warehouse(1, len(self.warehouses))
        self.warehouses.append(w)
        w = Warehouse(2, len(self.warehouses))
        self.warehouses.append(w)
        w = Warehouse(3, len(self.warehouses))
        self.warehouses.append(w)
        w = Warehouse(4, len(self.warehouses))
        self.warehouses.append(w)

    def BuildRandomWarehouses(self, totalLocations):
        for i in range(0, totalLocations):
            w = Location('W'+str(i), len(self.warehouses))
            self.warehouses.append(w)

    def BuildDistanceMatrix(self):
        totalLocs = len(self.locations)
        self.distanceMatrix = [[0 for i in range(0, totalLocs)] for j in range(0, totalLocs)]
        self.distanceMatrix[0][0] = 0
        self.distanceMatrix[1][1] = 0
        self.distanceMatrix[2][2] = 0
        self.distanceMatrix[3][3] = 0

        self.distanceMatrix[0][1] = 20.62;
        self.distanceMatrix[0][2] = 56.57;
        self.distanceMatrix[0][3] = 11.18;

        self.distanceMatrix[1][2] = 40.31;
        self.distanceMatrix[1][3] = 25.50;

        self.distanceMatrix[2][3] = 54.08;

        for i in range (0, len(self.distanceMatrix)):
            for j in range(i + 1, len(self.distanceMatrix)):
                self.distanceMatrix[j][i] = self.distanceMatrix[i][j];

    def BuildRandomDistanceMatrix(self, totalLocations):
        self.distanceMatrix = [[0 for i in range(0, totalLocations)] for j in range(0, totalLocations)]
        for i in range(0, len(self.distanceMatrix)):
            for j in range(i+1, len(self.distanceMatrix)):
                randomDist = 10 + random.randint(1, 20)
                self.distanceMatrix[i][j] = randomDist
                self.distanceMatrix[j][i] = randomDist

    def BuildCostMatrix(self):
        totalWarehouses = len(self.warehouses)
        self.costPerDistanceUnit = [[0 for i in range(0, totalWarehouses)] for j in range(0, totalWarehouses)]
        self.costPerDistanceUnit[0][0] = 0
        self.costPerDistanceUnit[1][1] = 0
        self.costPerDistanceUnit[2][2] = 0
        self.costPerDistanceUnit[3][3] = 0

        self.costPerDistanceUnit[0][1] = 8.5
        self.costPerDistanceUnit[0][2] = 1.5
        self.costPerDistanceUnit[0][3] = 5

        self.costPerDistanceUnit[1][2] = 0.5
        self.costPerDistanceUnit[1][3] = 0.5

        self.costPerDistanceUnit[2][3] = 2

        for i in range(0, len(self.costPerDistanceUnit)):
            for j in range(i + 1, len(self.costPerDistanceUnit)):
                self.costPerDistanceUnit[j][i] = self.costPerDistanceUnit[i][j];

    def BuildRandomCostMatrix(self,totalLocations):
        self.costPerDistanceUnit = [[0 for i in range(0, totalLocations)] for j in range(0, totalLocations)]
        for i in range(0, len(self.costPerDistanceUnit)):
            for j in range(i+1, len(self.costPerDistanceUnit)):
                randomPerDistanceUnit = 1 + random.randint(1, 4)
                self.costPerDistanceUnit[i][j] = randomPerDistanceUnit
                self.costPerDistanceUnit[j][i] = randomPerDistanceUnit
