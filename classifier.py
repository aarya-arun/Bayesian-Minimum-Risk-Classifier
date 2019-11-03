#!/usr/bin/env python3


import sys

import modelmanager as mm
import datamanager as dm


def main(argv):

    (trainNatFV, trainInterFV, trainFV,
     testNatFV, testInterFV, testFV) = dm.getData()

    normNat = mm.MultivariateNormal(trainNatFV)
    normInter = mm.MultivariateNormal(trainInterFV)

    out = []
    for x in testFV.transpose():
        if normNat.likelihood(x) > normInter.likelihood(x):
            out.append(1)
        else:
            out.append(0)

    print(out)
    print([1] * 4 + [0] * 6)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
