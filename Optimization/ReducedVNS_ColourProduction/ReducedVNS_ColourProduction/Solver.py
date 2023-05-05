from production_model import *
from random_generator import *


class Solution(object):
    def __init__(self):
        self.productionSequence = []
        self.cost = 0


class Solver():
    def __init__(self, m: ProductionModel):
        self.colours = m.colours
        self.setUpTimes = m.setUpTimes
        self.solution = None
        self.myRand = MyRandomGenerator(len(self.colours))

    def BuildExampleSolution(self):
        self.solution = Solution()
        self.solution.productionSequence.append(self.colours[4 - 1])
        self.solution.productionSequence.append(self.colours[3 - 1])
        self.solution.productionSequence.append(self.colours[1 - 1])
        self.solution.productionSequence.append(self.colours[2 - 1])
        self.solution.productionSequence.append(self.colours[5 - 1])
        self.solution.cost = self.calculateCostOfSolution(self.solution)

    def calculateCostOfSolution(self, solution):

        totalCost = 0

        for i in range(0, len(solution.productionSequence) - 1):
            a:Colour = solution.productionSequence[i]
            b:Colour = solution.productionSequence[i+1]
            totalCost = totalCost + self.setUpTimes[a.positionInTheList][b.positionInTheList]

        return totalCost

    def ReducedVNS(self):
        topLevelIterations = 1
        kmax = 2

        topLevelIterationsPerformed = 0

        while topLevelIterationsPerformed < topLevelIterations:
            topLevelIterationsPerformed += 1
            k = 1
            while k <= kmax:
                moveCost = 0

                if k == 1:
                    positionOfRemoved = self.myRand.positionForRemoval()
                    positionForReinsertion = self.myRand.positionForReinsertion(positionOfRemoved)
                    moveCost = self.CalculateCostForRelocationMove(positionOfRemoved, positionForReinsertion)
                    #moveCostNaive = self.CalculateCostForRelocationMoveNaive(positionOfRemoved, positionForReinsertion)
                    if moveCost < 0:
                        self.ApplyRelocationMove(positionOfRemoved, positionForReinsertion)
                        k = 1
                    else:
                        k = k + 1
                elif k == 2:
                    positionOfFirst = self.myRand.positionForFirstSwapped()
                    positionOfSecond = self.myRand.positionForSecondSwapped(positionOfFirst)
                    moveCost = self.CalculateCostForSwapMove(positionOfFirst, positionOfSecond)
                    #moveCostNaive = self.CalculateCostForRelocationMoveNaive(positionOfFirst, positionOfSecond)
                    if moveCost < 0:
                        self.ApplySwapMove(positionOfFirst, positionOfSecond)
                        k = 1
                    else:
                        k = k + 1

    def CalculateCostForRelocationMove(self, positionOfRemoved, positionForReinsertion):
        removalCost = self.CalculateRemovalCostForRelocation(positionOfRemoved)
        reinsertionCost = self.CalculateReinsertionCostForRelocation(positionOfRemoved, positionForReinsertion)
        return removalCost + reinsertionCost

    def CalculateRemovalCostForRelocation(self, positionOfRemoved):
        relocated = self.solution.productionSequence[positionOfRemoved]
        A:Colour = self.GetPredecessor(positionOfRemoved)
        B:Colour = self.GetSuccessor(positionOfRemoved)

        costAdded = 0
        if (A is not None and B is not None):
            costAdded = self.setUpTimes[A.positionInTheList][B.positionInTheList]

        costRemoved = 0
        if A is not None:
            costRemoved = self.setUpTimes[A.positionInTheList][relocated.positionInTheList]
        if B is not None:
            costRemoved += self.setUpTimes[relocated.positionInTheList][B.positionInTheList]

        return costAdded - costRemoved

    def CalculateReinsertionCostForRelocation(self, positionOfRemoved, positionForReinsertion):
        relocated = self.solution.productionSequence[positionOfRemoved]
        A = None
        B = None

        positionOfNewPred = None
        positionOfNewSucc = None

        if (positionForReinsertion > positionOfRemoved):
            positionOfNewPred = positionForReinsertion
            positionOfNewSucc = positionForReinsertion + 1
        else:
            positionOfNewPred = positionForReinsertion - 1
            positionOfNewSucc = positionForReinsertion

        if positionOfNewPred >= 0:
            A = self.solution.productionSequence[positionOfNewPred]
        if positionOfNewSucc <= len (self.solution.productionSequence) -1:
            B = self.solution.productionSequence[positionOfNewSucc]

        costRemoved = 0
        if (A is not None and B is not None):
            costRemoved = self.setUpTimes[A.positionInTheList][B.positionInTheList]

        costAdded = 0
        if (A is not None):
            costAdded = self.setUpTimes[A.positionInTheList][relocated.positionInTheList]
        if B is not None:
            costAdded += self.setUpTimes[relocated.positionInTheList][B.positionInTheList]

        return costAdded - costRemoved

    def CalculateCostForSwapMove(self, positionOfFirst, positionOfSecond):
        firstIndex = positionOfFirst
        secondIndex = positionOfSecond

        if firstIndex > secondIndex:
            firstIndex = positionOfSecond
            secondIndex = positionOfFirst

        firstColour = self.solution.productionSequence[firstIndex]
        secondColour = self.solution.productionSequence[secondIndex]

        predOfFirst = self.GetPredecessor(firstIndex)
        succOfFirst = self.GetSuccessor(firstIndex)

        predOfSecond = self.GetPredecessor(secondIndex)
        succOfSecond = self.GetSuccessor(secondIndex)

        costRemoved = 0
        costAdded = 0

        # the swapped are not in consecutive places
        if firstIndex != secondIndex - 1:

            if predOfFirst is not None:
                costRemoved += self.setUpTimes[predOfFirst.positionInTheList][firstColour.positionInTheList]
                costAdded += self.setUpTimes[predOfFirst.positionInTheList][secondColour.positionInTheList]

            costRemoved += self.setUpTimes[firstColour.positionInTheList][succOfFirst.positionInTheList]
            costAdded += self.setUpTimes[secondColour.positionInTheList][succOfFirst.positionInTheList]

            costRemoved += self.setUpTimes[predOfSecond.positionInTheList][secondColour.positionInTheList]
            costAdded += self.setUpTimes[predOfSecond.positionInTheList][firstColour.positionInTheList]

            if succOfSecond is not None:
                costRemoved += self.setUpTimes[secondColour.positionInTheList][succOfSecond.positionInTheList]
                costAdded += self.setUpTimes[firstColour.positionInTheList][succOfSecond.positionInTheList]
        #the swapped are in consecutive places
        else:

            if (predOfFirst is not None):
                costRemoved += self.setUpTimes[predOfFirst.positionInTheList][firstColour.positionInTheList]
                costAdded += self.setUpTimes[predOfFirst.positionInTheList][secondColour.positionInTheList]

            if (succOfSecond is not None):
                costRemoved += self.setUpTimes[secondColour.positionInTheList][succOfSecond.positionInTheList]
                costAdded += self.setUpTimes[firstColour.positionInTheList][succOfSecond.positionInTheList]

            costRemoved += self.setUpTimes[firstColour.positionInTheList][secondColour.positionInTheList]
            costAdded += self.setUpTimes[secondColour.positionInTheList][firstColour.positionInTheList]

        return costAdded - costRemoved


    def GetPredecessor(self, index):
        if (index > 0):
            return self.solution.productionSequence[index - 1]
        return None

    def GetSuccessor(self, index):
        if (index < len(self.solution.productionSequence) - 1):
            return self.solution.productionSequence[index + 1]
        return None

    def ApplyRelocationMove(self, positionOfRemoved, positionForReinsertion):
        relocated = self.solution.productionSequence
        del (self.solution.productionSequence[positionOfRemoved])
        self.solution.productionSequence.insert(positionForReinsertion, relocated)
        self.solution.cost = self.calculateCostOfSolution(self.solution)

    def ApplySwapMove(self, positionOfFirst, positionOfSecond):
        first = self.solution.productionSequence[positionOfFirst]
        second = self.solution.productionSequence[positionOfSecond]
        self.solution.productionSequence[positionOfFirst] = second
        self.solution.productionSequence[positionOfSecond] = first
        self.solution.cost = self.calculateCostOfSolution(self.solution)


