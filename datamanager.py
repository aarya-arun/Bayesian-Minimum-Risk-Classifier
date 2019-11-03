import numpy as np


def featureVector(fin):
    selfcites, NonLocalCount, Totalcites, NLIQ, OCQ, HINDEX, IC = [[], [], [], [], [], [], []]

    for i in fin:
        i = i.split(',')
        if(len(i) > 7):
            i = i[:-1]
        i = [float(x) for x in i[1:]]
        selfcites.append(i[0])
        NonLocalCount.append(i[1])
        Totalcites.append(i[2])
        NLIQ.append(i[3])
        OCQ.append(i[4])
        HINDEX.append(i[5])
        IC.append(i[6])

    # print(selfcites,NonLocalCount,Totalcites,NLIQ,OCQ,HINDEX,IC, sep="\n")
    features = np.array([np.array(i) for i in (selfcites, NonLocalCount, Totalcites,
                                                     NLIQ, OCQ, HINDEX, IC)])
    # print(features)
    return features

def getData(seed=0):
    np.random.seed(seed)
    finNat = open("Data/nat.csv", "r").readlines()
    finInter = open("Data/inter.csv", "r").readlines()
    np.random.shuffle(finNat)
    np.random.shuffle(finInter)

    finTest = finNat[:4] + finInter[:6]
    finNat = finNat[4:]
    finInter = finInter[6:]
    finTrain = finNat + finInter

    trainNatFV = featureVector(finNat)
    trainInterFV = featureVector(finInter)
    testFV = featureVector(finTest)
    trainFV = featureVector(finTrain)
    testNatFV = [] #testFV.transpose()[:4].transpose()
    testInterFV = [] #testFV.transpose()[4:].transpose()

    return (trainNatFV, trainInterFV, trainFV, testNatFV, testInterFV, testFV)
    
    # print()
    # printFeatureVector(fin)
