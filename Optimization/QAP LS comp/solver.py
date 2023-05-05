from qap import *


class Solution(object):
    
    def __init__(self):
        self.cost = 0
        self.assignedWarehouses = []


class SwapMove(object):
    def __init__(self):
        self.locationIndex1 = None
        self.locationIndex2 = None
        self.moveCost = None
        self.tentativeSolution = None

    def InitializeSwapMove(self):
        self.locationIndex1 = None
        self.locationIndex2 = None
        self.moveCost = 10 ** 9
        self.tentativeSolution = None


class Solver():
    def __init__(self, m: QuadraticAssignmentProblem):
        self.locations = m.locations
        self.warehouses = m.warehouses
        self.distanceMatrix = m.distanceMatrix
        self.costPerDistanceUnit = m.costPerDistanceUnit
        self.solution = Solution()

    def GenerateExampleSolution(self):
        w1 = self.warehouses[0]
        self.solution.assignedWarehouses.append(w1)
        w2 = self.warehouses[1]
        self.solution.assignedWarehouses.append(w2)
        w4 = self.warehouses[3]
        self.solution.assignedWarehouses.append(w4)
        w3 = self.warehouses[2]
        self.solution.assignedWarehouses.append(w3)

        self.solution.cost = self.CalculateCost(self.solution)

    def GenerateRandomSolution(self):
        for i in range(0, len(self.warehouses)):
            w1 = self.warehouses[i]
            self.solution.assignedWarehouses.append(w1)

        self.solution.cost = self.CalculateCost(self.solution)

    def CalculateCost(self, sol: Solution):
        totalCost = 0
        for i in range (0, len(sol.assignedWarehouses)):
            for j in range(0, len(sol.assignedWarehouses)):
                w1 = sol.assignedWarehouses[i]
                w2 = sol.assignedWarehouses[j]

                distanceBetweenLocations = self.distanceMatrix[i][j]
                costPerDistanceForThePair = self.costPerDistanceUnit[w1.indexInTheList][w2.indexInTheList]
                totalCost += distanceBetweenLocations * costPerDistanceForThePair

        return totalCost

    def Solve(self):
        #self.GenerateExampleSolution()
        self.GenerateRandomSolution()
        self.ReportSolution(self.solution)
        self.LocalSearch()
        self.ReportSolution(self.solution)

    def LocalSearch(self):
        sm = SwapMove()
        localOptimum = False
        #maxIter = 500
        #iter = 0
        
        self.CalculateCostComponents()
        while localOptimum == False:
            sm.InitializeSwapMove()
            self.FindBestSwapMove(sm)
            if sm.moveCost < 0:
                self.solution = sm.tentativeSolution
                print(self.solution.cost)
            else:
                localOptimum = True
            #iter = iter + 1

    def FindBestSwapMove(self, sm):
        for i in range(0, len(self.solution.assignedWarehouses)):
            for j in range(i + 1, len(self.solution.assignedWarehouses)):

                # Slow way
                sm.locationIndex1 = i
                sm.locationIndex2 = j
                candSol = self.FormNewSolutionWithSwap(sm)
                moveCost = candSol.cost - self.solution.cost

                # Fast way
                #moveCost = self.CalculateSwapMoveCost(i, j)

                # Faster way
                #moveCost = self.CalculateSwapMoveCostFaster(i, j)

                if moveCost < sm.moveCost:
                    sm.moveCost = moveCost
                    sm.locationIndex1 = i
                    sm.locationIndex2 = j
                    sm.tentativeSolution = self.FormNewSolutionWithSwap(sm)
        return sm

    def FormNewSolutionWithSwap(self, sm):
        newSol = Solution()
        newSol.assignedWarehouses = self.solution.assignedWarehouses.copy()
        newSol.assignedWarehouses[sm.locationIndex1] = self.solution.assignedWarehouses[sm.locationIndex2]
        newSol.assignedWarehouses[sm.locationIndex2] = self.solution.assignedWarehouses[sm.locationIndex1]
        newSol.cost = self.CalculateCost(newSol)

        self.CalculateCostComponents()

        return newSol

    def CalculateSwapMoveCost(self, k, l):
        costRemoved = 0
        costAdded = 0

        rem1 = self.solution.assignedWarehouses[k]
        rem2 = self.solution.assignedWarehouses[l]

        # All flows coming to and going  from rem1
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem1.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[k][i]
                costRemoved += costPerDist * dist

                costPerDist =  self.costPerDistanceUnit[other.indexInTheList][rem1.indexInTheList]
                dist = self.distanceMatrix[i][k]
                costRemoved += costPerDist * dist

        # All flows coming to and going to rem2
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem2.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[l][i]
                costRemoved += costPerDist * dist

                costPerDist = self.costPerDistanceUnit[other.indexInTheList][rem2.indexInTheList]
                dist = self.distanceMatrix[i][l]
                costRemoved += costPerDist * dist

        costRemoved += self.costPerDistanceUnit[rem1.indexInTheList][rem2.indexInTheList] * self.distanceMatrix[k][l]
        costRemoved += self.costPerDistanceUnit[rem2.indexInTheList][rem1.indexInTheList] * self.distanceMatrix[k][l]

        # All flows coming to and going to the new rem1
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem2.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[k][i]
                costAdded += costPerDist * dist

                costPerDist = self.costPerDistanceUnit[other.indexInTheList][rem2.indexInTheList]
                dist = self.distanceMatrix[i][k]
                costAdded += costPerDist * dist

        # All flows coming to and going to the new rem2
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem1.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[l][i]
                costAdded += costPerDist * dist

                costPerDist = self.costPerDistanceUnit[other.indexInTheList][rem1.indexInTheList]
                dist = self.distanceMatrix[i][l]
                costAdded += costPerDist * dist

        costAdded += self.costPerDistanceUnit[rem2.indexInTheList][rem1.indexInTheList] * self.distanceMatrix[k][l]
        costAdded += self.costPerDistanceUnit[rem1.indexInTheList][rem2.indexInTheList] * self.distanceMatrix[l][k]
        totalCost = costAdded - costRemoved

        return totalCost

    def CalculateSwapMoveCostFaster(self, k, l):
        costRemoved = 0
        costAdded = 0

        rem1 = self.solution.assignedWarehouses[k]
        rem2 = self.solution.assignedWarehouses[l]

        #All flows coming to and going from rem1
        costRemoved = rem1.costComponent
        costRemoved += rem2.costComponent
        #The following cost has been taken into consideration twice because of the cost components
        costRemoved -= self.costPerDistanceUnit[rem1.indexInTheList][rem2.indexInTheList] * self.distanceMatrix[k][l]
        costRemoved -= self.costPerDistanceUnit[rem2.indexInTheList][rem1.indexInTheList] * self.distanceMatrix[l][k]

        # All flows coming to and going to the new rem1
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem2.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[k][i]
                costAdded += costPerDist * dist

                costPerDist = self.costPerDistanceUnit[other.indexInTheList][rem2.indexInTheList]
                dist = self.distanceMatrix[i][k]
                costAdded += costPerDist * dist

        # All flows coming to and going to the new rem2
        for i in range(0, len(self.solution.assignedWarehouses)):
            other = self.solution.assignedWarehouses[i]
            if i != k and i != l:
                costPerDist = self.costPerDistanceUnit[rem1.indexInTheList][other.indexInTheList]
                dist = self.distanceMatrix[l][i]
                costAdded += costPerDist * dist

                costPerDist = self.costPerDistanceUnit[other.indexInTheList][rem1.indexInTheList]
                dist = self.distanceMatrix[i][l]
                costAdded += costPerDist * dist

        costAdded += self.costPerDistanceUnit[rem2.indexInTheList][rem1.indexInTheList] * self.distanceMatrix[k][l]
        costAdded += self.costPerDistanceUnit[rem1.indexInTheList][rem2.indexInTheList] * self.distanceMatrix[l][k]
        totalCost = costAdded - costRemoved

        return totalCost


    def ReportSolution(self, solution):
        for i in range(0, len(solution.assignedWarehouses)):
            print(solution.assignedWarehouses[i].indexInTheList, end=' ')
        print(solution.cost)

    def CalculateCostComponents(self):
        for i in range(0, len(self.solution.assignedWarehouses)):
            w = self.solution.assignedWarehouses[i]
            w.costComponent = 0

            for j in range(0, len(self.solution.assignedWarehouses)):
                other = self.solution.assignedWarehouses[j]
                w.costComponent += self.costPerDistanceUnit[w.indexInTheList][other.indexInTheList] * self.distanceMatrix[i][j]
                w.costComponent += self.costPerDistanceUnit[other.indexInTheList][w.indexInTheList] * self.distanceMatrix[j][i]