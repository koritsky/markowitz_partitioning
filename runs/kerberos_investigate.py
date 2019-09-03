from __future__ import print_function

import sys

import dimod
from hybrid.reference.kerberos import KerberosSampler



energy_threshold = None
if len(sys.argv) > 2:
    energy_threshold = float(sys.argv[2])

solution = KerberosSampler().sample(bqm, max_iter=10, convergence=3,
                                    energy_threshold=energy_threshold)