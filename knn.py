import sys
import pandas as pd
import time
import numpy as np
import shutil

def knnTrain(trainfile,modelfile):
    print "TRAINING......."
    shutil.copy(trainfile, modelfile)
    print "TRAINING COMPLETED: MODEL FILE GENERATED"

def knnTest(testfile,modelfile):
    print "READING......."
    train = pd.read_table(modelfile, sep='\s+', header=None)
    test = pd.read_table(testfile, sep='\s+', header=None)

    trainorient=train.iloc[:][1].values
    testid=test.iloc[:][0].values
    testorient=test.iloc[:][1].values

    del train[0]
    del train[1]
    del test[0]
    del test[1]

    train.columns = range(train.shape[1])
    testarray = test.values
    trainarray = train.values
    print "FILES READ........."

    knnCompute(testarray,trainarray,trainorient,testorient,testid)

def knnCompute(testarray,trainarray,trainorient,testorient,testid):
    print "COMPUTING KNN........"
    solutionset = []
    for i in range(0, len(testarray)):
        testvector = testarray[i]
        t = []
        distancelist = []
        for trainvector in trainarray:
            distance = (np.sum(np.power((trainvector - testvector), 2)))
            distancelist.append(distance)
        t = tuple(zip(distancelist, trainorient))
        k = 29
        tups_deg = []
        B = sorted(t, key=lambda x: x[0])
        # print B
        for j in range(0, k):
            tups_deg.append(B[j][1])
        mode = max(set(tups_deg), key=tups_deg.count)
        solutionset.append(mode)
    knnwriteFile(solutionset,testid,testorient)

def knnwriteFile(solutionset,testid,testorient):
    s = []
    s = list(zip(testid, solutionset))

    a = np.array(solutionset)
    b = np.array(testorient)
    count = np.sum(a == b)
    percent = (count / float(len(testorient))) * 100.0
    print "Percentage correct: {}".format(round(percent,2))
    with open('output.txt', 'w') as fp:
        fp.write('\n'.join('%s %s' % tups for tups in s))




option=sys.argv[1]
optionfile=sys.argv[2]
modelfile=sys.argv[3]
model=sys.argv[4]

start_time=time.time()
if option=="train":
    knnTrain(optionfile,modelfile)
elif option=="test":
    knnTest(optionfile,modelfile)
print "Time Taken: {} minutes".format((time.time()-start_time)/60.0)