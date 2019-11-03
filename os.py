import os
import math

for i in range(math.factorial(22)*math.factorial(20)):
    os.system("py -3 classifier.py "+str(i)+" >> out.txt")