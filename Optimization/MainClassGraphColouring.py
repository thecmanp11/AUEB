
class Colour(object):
    def __init__(self, idd):
        self.id = idd
        self.forbiddenNodes = []
        self.assignedNodes = []
        self.forbiddenNodeSet = set()

    def AddForbidenNodes(self, node):
        for n in node.forbiddenNodes:

            #list implementation slower "in" function: goes through every element to look for the one we are searching for
            if (n in self.forbiddenNodes) == False:
                self.forbiddenNodes.append(n)

            #set implementation faster in function: uses hashes to quicly identify the element we are looking for
            if (n in self.forbiddenNodeSet) == False:
                self.forbiddenNodeSet.add(n)


class Node(object):
    def __init__(self, idd):
        self.id = idd
        self.forbiddenNodes = []
        self.assignedColour = None


def SetUpForbiddenNodes(nodes, id, listWithForbiddenIds):
    n = nodes[id-1]
    #n.forbiddenNodes = [nodes[i-1] for i in listWithForbiddenIds] #faster coding
    for i in range(0, len(listWithForbiddenIds)):
        idOfForbidden = listWithForbiddenIds[i]
        n.forbiddenNodes.append(nodes[idOfForbidden - 1])


def BuildInput(nodes):
    for i in range(1,16):
        n = Node(i)
        nodes.append(n)

    SetUpForbiddenNodes(nodes, 1, [5, 6])
    SetUpForbiddenNodes(nodes, 2, [3, 5])
    SetUpForbiddenNodes(nodes, 3, [2, 6, 7])
    SetUpForbiddenNodes(nodes, 4, [7])
    SetUpForbiddenNodes(nodes, 5, [1, 2, 6, 8, 9])
    SetUpForbiddenNodes(nodes, 6, [1, 3, 5, 9, 10])
    SetUpForbiddenNodes(nodes, 7, [3, 4, 10, 11, 12])
    SetUpForbiddenNodes(nodes, 8, [5])
    SetUpForbiddenNodes(nodes, 9, [5, 6, 10])
    SetUpForbiddenNodes(nodes, 10, [6, 7, 9])
    SetUpForbiddenNodes(nodes, 11, [7])
    SetUpForbiddenNodes(nodes, 12, [7, 14])
    SetUpForbiddenNodes(nodes, 13, [14, 15])
    SetUpForbiddenNodes(nodes, 14, [12, 13, 15])
    SetUpForbiddenNodes(nodes, 15, [13, 14])




#stable ordering for tie breakers
#first sor with the tie breaker and then with the most important criterion - the order if ties occur is preservred = stable ordering
#an hierarchical function could also be written
#nodesToBeSorted2 = nodes.copy()
#nodesToBeSorted2.sort(key = lambda x : -x.id)
#nodesToBeSorted2.sort(key = lambda x : -len(x.forbiddenNodes))

def FindFirstAvailableColour(node, coloursUsed):
    for c in coloursUsed:
        if (node in c.forbiddenNodes) == False:
            return c
        # set implementation is faster
        if (node in c.forbiddenNodeSet) == False:
            return c
    c = Colour(len(coloursUsed) + 1)
    coloursUsed.append(c)
    return c




def SolveProblem(sortedNodes, coloursUsed):
    for node in sortedNodes:
        colourSelected:Colour = FindFirstAvailableColour(node, coloursUsed)
        node.assignedColour = colourSelected
        colourSelected.assignedNodes.append(node)
        colourSelected.AddForbidenNodes(node)

# execution starts here
nodes = []
coloursUsed = []
BuildInput(nodes)
sortedNodes = nodes.copy()
sortedNodes.sort(key = lambda x : -len(x.forbiddenNodes))
SolveProblem(sortedNodes, coloursUsed)

