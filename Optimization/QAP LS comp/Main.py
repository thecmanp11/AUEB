from solver import *
import timeit


m = QuadraticAssignmentProblem()
# m.BuildExampleModel()
m.BuildRandomModel()
s = Solver(m)
start = timeit.default_timer()
s.Solve()

stop = timeit.default_timer()
print('\nTime: ', stop - start)
