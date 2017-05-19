import sys
import math
import random

# python net_arch.py <arch>

arch = map(int, sys.argv[1].split(","))

nlayers = len(arch)

for i in range(nlayers - 1):
    for j in range((arch[i] + 1) * arch[i + 1]):
        print random.uniform(-1.0/math.sqrt(arch[i]), 1.0/math.sqrt(arch[i]))
