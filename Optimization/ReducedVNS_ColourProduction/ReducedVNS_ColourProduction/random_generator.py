class MyRandomGenerator:
    def __init__(self, num):

        self.lastIndexUsed = 0
        self.removalOptions = num
        self.reinsertionOptions = num - 1
        self.firstSwapOptions = num
        self.secondSwapOptions = num - 1
        self.epsilon = 0.00001

        self.randomNumbers = []
        self.randomNumbers.append(0.689)
        self.randomNumbers.append(0.804)
        self.randomNumbers.append(0.452)
        self.randomNumbers.append(0.852)
        self.randomNumbers.append(0.492)
        self.randomNumbers.append(0.652)
        self.randomNumbers.append(0.196)
        self.randomNumbers.append(0.152)
        self.randomNumbers.append(0.830)
        self.randomNumbers.append(0.400)

    def positionForRemoval(self):
        randomNumber = self.randomNumbers[self.lastIndexUsed]
        self.lastIndexUsed += 1
        denominator = 1.0 / self.removalOptions
        result = randomNumber / denominator - self.epsilon
        finalResult = int(result)
        return finalResult

    def positionForReinsertion(self, positionOfRemoved):
        randomNumber = self.randomNumbers[self.lastIndexUsed]
        self.lastIndexUsed += 1
        denominator = 1.0 / self.reinsertionOptions
        result = randomNumber / denominator - self.epsilon
        finalResult = int(result)
        if finalResult < positionOfRemoved:
            return finalResult
        else:
            return finalResult + 1

    def positionForFirstSwapped(self):
        randomNumber = self.randomNumbers[self.lastIndexUsed]
        self.lastIndexUsed += 1
        denominator = 1.0 / self.firstSwapOptions
        result = randomNumber / denominator - self.epsilon
        finalResult = int(result)
        return finalResult

    def positionForSecondSwapped(self, positionOfFirst):
        randomNumber = self.randomNumbers[self.lastIndexUsed]
        self.lastIndexUsed += 1
        denominator = 1.0 / self.secondSwapOptions
        result = randomNumber / denominator - self.epsilon
        finalResult = int(result)
        if finalResult < positionOfFirst:
            return finalResult
        else:
            return finalResult + 1