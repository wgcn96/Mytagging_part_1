
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def loadDataFromTxt(fileName):
    dataSet=np.mat(np.loadtxt(fileName,delimiter='\t'))
    return dataSet

def normalization(dataSet):
    colNum=np.shape(dataSet)[1]
    for index in range(colNum):
        colMax=np.max(dataSet[:,index])
        dataSet[:,index]=dataSet[:,index]/colMax
    return dataSet

def initWithFuzzyMat(n,k):
    fuzzyMat=np.mat(np.zeros((k,n)))
    for colIndex in range(n):
        memDegreeSum=0
        randoms=np.random.rand(k-1,1)
        for rowIndex in range(k-1):
            fuzzyMat[rowIndex,colIndex]=randoms[rowIndex,0]*(1-memDegreeSum)
            memDegreeSum+=fuzzyMat[rowIndex,colIndex]
        fuzzyMat[-1,colIndex]=1-memDegreeSum
    return fuzzyMat

def eculidDistance(vectA,vectB):
    return np.sqrt(np.sum(np.power(vectA-vectB,2)))

def calCentWithFuzzyMat(dataSet,fuzzyMat,p):
    n,m=dataSet.shape
    k=fuzzyMat.shape[0]
    centroids=np.mat(np.zeros((k,m)))
    for rowIndex in range(k):
        degExpArray=np.power(fuzzyMat[rowIndex,:],p)
        denominator=np.sum(degExpArray)
        numerator=np.array(np.zeros((1,m)))
        for colIndex in range(n):
            numerator+=dataSet[colIndex]*degExpArray[0,colIndex]
        centroids[rowIndex,:]=numerator/denominator
    return centroids

def calFuzzyMatWithCent(dataSet,centroids,p):
    n,m=dataSet.shape
    c=centroids.shape[0]
    fuzzyMat=np.mat(np.zeros((c,n)))
    for rowIndex in range(c):
        for colIndex in range(n):
            d_ij=eculidDistance(centroids[rowIndex,:],dataSet[colIndex,:])
            fuzzyMat[rowIndex,colIndex]=1/np.sum([np.power(d_ij/eculidDistance(centroid,dataSet[colIndex,:]),2/(p-1)) for centroid in centroids])
    return fuzzyMat

def calTargetFunc(dataSet,fuzzyMat,centroids,k,p):
    n,m=dataSet.shape
    c=fuzzyMat.shape[0]
    targetFunc=0
    for rowIndex in range(c):
        for colIndex in range(n):
            targetFunc+=eculidDistance(centroids[rowIndex,:],dataSet[colIndex,:])**2*np.power(fuzzyMat[rowIndex,colIndex],p)
    return targetFunc

def fuzzyCMean(dataSet,k,p,initMethod=initWithFuzzyMat):
    n,m=dataSet.shape
    fuzzyMat=initWithFuzzyMat(n,k)
    centroids=calCentWithFuzzyMat(dataSet,fuzzyMat,p)
    lastTargetFunc=calTargetFunc(dataSet,fuzzyMat,centroids,k,p)

    fuzzyMat=calFuzzyMatWithCent(dataSet,centroids,p)
    centroids=calCentWithFuzzyMat(dataSet,fuzzyMat,p)
    targetFunc=calTargetFunc(dataSet,fuzzyMat,centroids,k,p)
    while lastTargetFunc*0.99>targetFunc:
        lastTargetFunc=targetFunc
        fuzzyMat=calFuzzyMatWithCent(dataSet,centroids,p)
        centroids=calCentWithFuzzyMat(dataSet,fuzzyMat,p)
        targetFunc=calTargetFunc(dataSet,fuzzyMat,centroids,k,p)
    return fuzzyMat,centroids


if __name__=='__main__':
    iris_info = load_iris()
    # info(iris_info)
    dataSet = iris_info.data[:, :4]
    dataSet=normalization(dataSet)
    fuzzyMat,centroids=fuzzyCMean(dataSet,3,2)
    print('fuzzyMat=\n',fuzzyMat)
    print(np.sum(fuzzyMat,axis=0))
    print('centroids=\n',centroids)
