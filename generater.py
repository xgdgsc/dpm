#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np,argparse

def gaussianClusterMixture(clusterNumber,centerMin,centerMax,var,pointNum):
    pointArrayList=[]
    cov=var*np.identity(2)
    for i in range(clusterNumber):
        centerX=np.random.random_integers(centerMin,centerMax)
        centerY=np.random.random_integers(centerMin,centerMax)
        print "("+str(centerX)+','+str(centerY)+")"
        gCluster=np.random.multivariate_normal((centerX,centerY),cov,pointNum)
        #print gCluster
        pointArrayList.append(gCluster)

    points=pointArrayList[0]
    for pointArray in pointArrayList[1:]:
        points=np.vstack([points,pointArray])

    return points




if __name__=="__main__":
    parser = argparse.ArgumentParser("Gaussian clusters generating routine.")
    parser.add_argument('-nc','--num_clusters',help='Number of clusters',required=False)
    parser.add_argument('-np','--num_points',help='Number of points in a cluster',required=False)
    parser.add_argument('-l','--center_lower',help='Lower bound for generating random gaussian mean for both X and Y axis',required=False)
    parser.add_argument('-u','--center_upper',help='Upper bound for generating random gaussian mean for both X and Y axis',required=False)
    parser.add_argument('-v','--var',help='Variance for gaussian',required=False)
    parser.add_argument('-o','--out_file',help='Path for outfile',required=False)
    args=parser.parse_args()
    if not args.num_clusters:
        args.num_clusters=3
    if not args.num_points:
        args.num_points=100
    if not args.center_lower:
        args.center_lower=-5
    if not args.center_upper:
        args.center_upper=5
    if not args.var:
        args.var=1
    if not args.out_file:
        args.out_file="gClusters.txt"
    pointArray=gaussianClusterMixture(int(args.num_clusters),args.center_lower,args.center_upper,args.var,args.num_points)
    #print pointArray
    np.savetxt(args.out_file,pointArray)
