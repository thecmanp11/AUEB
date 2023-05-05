from Solver import *

m = ProductionModel()
m.BuildExampleModel()
s = Solver(m)
s.BuildExampleSolution()
s.ReducedVNS()
