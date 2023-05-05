class ItemType():
    def __init__(self, str, numberOfItems, len, pr, minInShelf, maxInShelf):
        self.typeName = str
        self.numberOfItems = numberOfItems
        self.length = len
        self.profitPerItem = pr
        self.minimumNumberInOneShelf = minInShelf
        self.maximumNumberInOneShelf = maxInShelf


class Shelf():
    def __init__(self, k):
        self.length = k
        self.residualLength = self.length
        self.insertedItems = []


class Item(object):
    def __init__(self, tp: ItemType):
        self.profit = tp.profitPerItem
        self.length = tp.length
        self.itemType = tp
        self.isInserted = False

class ShelfAllocator:
    def __init__(self):
        self.itemTypes = []
        self.shelves = []
        self.items = []
        self.length = 0
        self.totalProfit = 0

    def GenerateCompleteItemList(self):

        for i in range (0, len(self.itemTypes)):
            tp:ItemType = self.itemTypes[i]
            for j in range (0, tp.numberOfItems):
                it = Item(tp)
                it.ID = len(self.items)
                self.items.append(it)

        self.items.sort(key=lambda x: x.profit/x.length, reverse=True)


    def BuildExampleInput(self, shelfLength):
        tp = ItemType("No.1", 3, 20, 100, 0, 2)
        self.itemTypes.append(tp)
        tp = ItemType("No.2", 4, 40, 800, 0, 2)
        self.itemTypes.append(tp)
        tp = ItemType("No.3", 7, 30, 300, 0, 4)
        self.itemTypes.append(tp)
        tp = ItemType("No.4", 2, 50, 700, 0, 2)
        self.itemTypes.append(tp)
        tp = ItemType("No.5", 5, 20, 200, 0, 4)
        self.itemTypes.append(tp)
        tp = ItemType("No.6", 4, 50, 400, 0, 3)
        self.itemTypes.append(tp)

        s1 = Shelf(shelfLength)
        self.shelves.append(s1)
        s2 = Shelf(shelfLength)
        self.shelves.append(s2)

    def Solve(self):
        self.GenerateCompleteItemList()
        self.PushIterativelyInTheSolution()

    def PushIterativelyInTheSolution(self):
        self.totalProfit = 0

        for i in range(0, len(self.items)):
            it = self.items[i]
            indexOfFeasibleShelf = self.IndexOfShelfForFeasibleInsertion(it)

            if indexOfFeasibleShelf != -1:
                shelf = self.shelves[indexOfFeasibleShelf]
                shelf.insertedItems.append(it)
                shelf.residualLength = shelf.residualLength - it.length
                self.totalProfit = self.totalProfit + it.profit

    def IndexOfShelfForFeasibleInsertion(self, it: Item):

        for i in range(0, len(self.shelves)):
            sh = self.shelves[i]
            if it.length > sh.residualLength:
                continue

            #generator use
            #countOfSameTypeAlreadyInShelf = sum(1 for f in sh.insertedItems if f.itemType == it.itemType)
            countOfSameTypeAlreadyInShelf = self.CountSameTypeObjectsInShelf(sh, it)

            if countOfSameTypeAlreadyInShelf == it.itemType.maximumNumberInOneShelf:
                continue

            return i

        return -1

    def CountSameTypeObjectsInShelf(self, sh:Shelf, it):
        occur = 0
        for i in range (0, len(sh.insertedItems)):
            inserted = sh.insertedItems[i]
            if inserted.itemType == it.itemType:
                occur = occur + 1
        return occur