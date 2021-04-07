import numpy as np

def getBatch(data, testMode, epochs=3):
    data = data.to_numpy()
    batchSize = 3
    # how many hours in the past are we using to predict?
    lookbackPeriods = 6
    # how many hours in the future are we predicing?
    forwardPeriods = 1
    endIdx = len(data)-50
    count = 0
    while count < epochs:
        count += 1
        print("\nepoch # ", count)
        rowIdx = 0
        while (rowIdx < endIdx):
            featureList = []
            labelList = []
            for _ in range(batchSize):
                featureList.append(data[rowIdx:rowIdx+lookbackPeriods])
                labelList.append(data[rowIdx+lookbackPeriods+forwardPeriods-1][-1])
                rowIdx += 1

            features = np.array(featureList)
            features = np.reshape(features, newshape=[batchSize, lookbackPeriods, -1])
            labels = np.array(labelList)
            labels = np.reshape(labels, newshape=[batchSize, 1, -1])
            if testMode:
                yield features
            else:
                yield (features, labels)