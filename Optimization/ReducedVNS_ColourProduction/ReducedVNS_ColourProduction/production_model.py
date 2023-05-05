class Colour(object):
    def __init__(self, i):
        self.positionInTheList = i
        self.ID = i + 1



class ProductionModel:
    def __init__(self):
        self.colours = []
        self.setUpTimes = []

    def BuildExampleModel(self):
        c = Colour(len(self.colours))
        self.colours.append(c)
        c = Colour(len(self.colours))
        self.colours.append(c)
        c = Colour(len(self.colours))
        self.colours.append(c)
        c = Colour(len(self.colours))
        self.colours.append(c)
        c = Colour(len(self.colours))
        self.colours.append(c)

        totalColours = len(self.colours)
        self.setUpTimes = [[0 for j in range(0, totalColours)] for i in range(0, totalColours)]

        for i in range (0, totalColours):
            self.setUpTimes[i][i] = 0.0

            self.setUpTimes[0][1] = 56.92
            self.setUpTimes[0][2] = 70.45
            self.setUpTimes[0][3] = 36.24
            self.setUpTimes[0][4] = 68.15

            self.setUpTimes[1][2] = 18.87
            self.setUpTimes[1][3] = 72.37
            self.setUpTimes[1][4] = 12.04

            self.setUpTimes[2][3] = 90.25
            self.setUpTimes[2][4] = 19.31

            self.setUpTimes[3][4] = 80.22

        for i in range (0, totalColours):
            for j in range (i + 1, totalColours):
                self.setUpTimes[j][i] = self.setUpTimes[i][j]





