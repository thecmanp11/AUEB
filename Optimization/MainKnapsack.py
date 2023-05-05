import random

class Knapsack:
    def __init__(self):
        self.items = []
        self.capacity = 0

class Item:
    def __init__(self):
        self.profit = 0
        self.weight = 0
        self.isInserted = False


class Solution:
    def __init__(self):
        self.insertedItems = []
        self.selectionMatrix = []
        self.totalProfit = 0
        self.totalWeight = 0

def buildInputRandomly(k, numberOfItems, minProfit, maxProfit, minWeight, maxWeight, capacity):
    random.seed(1)
    k.capacity = capacity
    for i in range(0, numberOfItems):
        print(type(i))
        it = Item()
        it.profit = minProfit + random.randint(0, maxProfit - minProfit)
        it.weight = minWeight + random.randint(0, maxWeight - minWeight)
        it.ID = i
        k.items.append(it)

def solve(ks):

    s = Solution()
    s.selectionMatrix = [False for x in range(len(ks.items))]
    executionCondition = True

    while (executionCondition == True):

        bestCriterionValue = -1
        positionOfInsertedItem = -1

        for i in range(0, len(ks.items)):

            candidateForInsertion: Item = ks.items[i]

            if candidateForInsertion.isInserted == True:
                continue

            if s.totalWeight + candidateForInsertion.weight > ks.capacity:
                continue

            criterion = candidateForInsertion.profit
            #criterion = candidateForInsertion.profit / candidateForInsertion.weight
            if criterion > bestCriterionValue:
                bestCriterionValue = criterion
                positionOfInsertedItem = i

        if positionOfInsertedItem == -1:
            executionCondition = False
        else:
            insertedItem = k.items[positionOfInsertedItem]
            s.insertedItems.append(insertedItem)
            s.selectionMatrix[insertedItem.ID] = True
            s.totalWeight = s.totalWeight + insertedItem.weight
            s.totalProfit = s.totalProfit + insertedItem.profit
            insertedItem.isInserted = True

    return s

def solve_alternativeImplementation(ks: Knapsack):
    s = Solution()
    s.selectionMatrix = [False for x in range(len(ks.items))]

    sortedItems = ks.items.copy()
    sortedItems.sort(key=lambda x: x.profit, reverse=True)

    for i in range(0, len(sortedItems)):
        candidateItem = sortedItems[i]
        if s.totalWeight + candidateItem.weight <= ks.capacity:
            s.insertedItems.append(candidateItem)
            s.selectionMatrix[candidateItem.ID] = True
            s.totalWeight = s.totalWeight + candidateItem.weight
            s.totalProfit = s.totalProfit + candidateItem.profit

    return s

# code execution starts here
numberOfItems = 1000
minProfit = 1000
maxProfit = 2000
minWeight = 30
maxWeight = 40
capacity = 500
k = Knapsack()
buildInputRandomly(k, numberOfItems, minProfit, maxProfit, minWeight, maxWeight, capacity)
s = solve_alternativeImplementation(k)
print (s.totalWeight)
print(s.totalProfit)

