#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np,argparse,random,matplotlib.pyplot as plt

class gibbs(object):

    def __init__(self,points):
        self.points=points
        self.N=len(points)
        self.K=np.random.random_integers(1,self.N/10)
        self.z=np.random.random_integers(1,self.K,size=(self.N,))
        self.Phi=np.random.rand(self.K,2)*10-5
        self.cov=np.identity(2)
        self.Cluster_num_List=[]
        self.D_K_alpha_List=[]
        self.D_M1_List=[]
        self.D_M2_List=[]
        self.Ep0=0
        self.iterList=[]

    def sampling(self,maxIter,alpha):
        for i in range(1,self.N):
            self.Ep0+=alpha/(i-1+alpha)
        self.iterList=range(1,maxIter+1)
        d=2
        sigI=np.linalg.inv(self.cov)
        sig=self.cov
        sigDet=np.linalg.det(sig)
        sigma=1
        sigPrime=np.linalg.inv(np.identity(2)/sigma**2+sigI)
        sigPrimeDet=np.linalg.det(sigPrime)
        pi=np.pi
        n=self.N
        sig_k1_prime=np.linalg.inv(np.identity(2)/sigma**2+1*sigI)
        #print "K:"+str(self.K)
        #print "len Phi:"+str(len(self.Phi))
        for k in range(1,self.K+1):
            c_k,sum_x=self.n_k_s(k)
            sig_k_prime=np.linalg.inv(np.identity(2)/sigma**2+c_k*sigI)
            mu_k_prime=sig_k_prime.dot(sigI.dot(sum_x))
            #print mu_k_prime
            self.Phi[k-1]=np.random.multivariate_normal(mu_k_prime,sig_k_prime)

        for iterator in range(maxIter):
            print "\nIter:"+str(iterator)
            #sample zi
            for zi in range(self.N):
                pList=[]
                for k in range(1,self.K+1):

                    pzi_k=self.n_k_i(k,zi)*np.exp(-0.5*(self.points[zi]-self.Phi[k-1])[np.newaxis].dot(sigI).dot(self.points[zi]-self.Phi[k-1])[np.newaxis].T)/((n-1+alpha)*((2*pi)**(d/2))*sigDet**0.5)
                    pList.append(pzi_k)

                pzi_k_1=alpha*(sigPrimeDet**0.5)*np.exp(0.5*self.points[zi][np.newaxis].dot(sigI.dot(sigPrime).dot(sigI)-sigI).dot(self.points[zi][np.newaxis].T))/((n-1+alpha)*((2*pi)**(d/2))*(sigma**d)*(sigDet**0.5))
                pList.append(pzi_k_1)
                #print sum(pList)
                # print "pList:"
                # print pList
                kList=range(1,self.K+2)
                sampleList=zip(kList,pList)
                #print sampleList
                k=weighted_choice(sampleList)
                #print "k:"+str(k)
                self.z[zi]=k
                if k==self.K+1:
                    self.K+=1
                    mu_k_prime=sig_k1_prime.dot(sigI.dot(self.points[zi]))
                    self.Phi=np.vstack((self.Phi,np.random.multivariate_normal(mu_k_prime,sig_k1_prime)))
                # else:
                #     c_k,sum_x=self.n_k_s(k)
                #     sig_k_prime=np.linalg.inv(np.identity(2)/sigma**2+c_k*sigI)
                #     mu_k_prime=sig_k_prime.dot((sigI.dot(sum_x)))
                #     self.Phi[k-1]=np.random.multivariate_normal(mu_k_prime,sig_k_prime)

            clusterCount=0
            zeroList=[]
            for k in range(1,self.K+1):
                if self.n_k(k)!=0:
                    clusterCount+=1

                else:
                    zeroList.append(k-1)

            origKList=range(1,self.K+1)
            kDict={}
            for k in origKList:
                count=0
                for kz in zeroList:
                    if k>kz+1:
                        count+=1
                    elif k==kz+1:
                        count=k
                        break
                    elif k<kz+1:
                        break
                kDict[k]=k-count
                #print map(lambda x:x+1,zeroList)
                #print kDict

            for zi in range(self.N):
                self.z[zi]=kDict[self.z[zi]]

                #print self.z

            self.Phi=np.delete(self.Phi,zeroList,0)
            self.K=self.K-len(zeroList)


            print "cluster:"+str(clusterCount)
            print "K:"+str(self.K)
            print "len Phi:"+str(len(self.Phi))

            #sample Phi_k
            for k in range(1,self.K+1):
                c_k,sum_x=self.n_k_s(k)
                sig_k_prime=np.linalg.inv(np.identity(2)/sigma**2+c_k*sigI)
                mu_k_prime=sig_k_prime.dot(sigI.dot(sum_x))
                #print mu_k_prime
                self.Phi[k-1]=np.random.multivariate_normal(mu_k_prime,sig_k_prime)
                #print self.Phi[k-1]
            print self.Phi
            print ''
            self.calcDist(clusterCount,sigma,sig,sigI)

    #record dist in iterations
    def calcDist(self,clusterCount,sigma,sig,sigI):
        #append cluster number
        self.Cluster_num_List.append(clusterCount)

        #append D_k_alpha
        D_k_alpha=clusterCount-self.Ep0
        self.D_K_alpha_List.append(D_k_alpha)

        #append D_M1
        D_M1=0
        for k in range(self.K):

            det=self.Phi[k].dot(self.Phi[k])
            D_M1+=det**0.5/(sigma**2)
        self.D_M1_List.append(D_M1)

        #append D_M2
        D_M2=0
        for zi in range(self.N):
            D_M2+=np.linalg.det((self.points[zi]-self.Phi[self.z[zi]-1])[np.newaxis].dot(sigI).dot(self.points[zi]-self.Phi[self.z[zi]-1])[np.newaxis].T)**0.5
        self.D_M2_List.append(D_M2)

    def n_k(self,k):
        nk=0
        for zk in self.z:
            if k==zk:
                nk+=1
        return nk

    def n_k_i(self,k,i):
        nk=0
        for zi in range(self.N):
            if k==self.z[zi] and i!=zi:
                nk+=1
        return nk

    def n_k_s(self,k):
        nk=0
        sum_x=np.zeros(2)
        for zi in range(self.N):
            if k==self.z[zi]:
                nk+=1
                sum_x=np.add(sum_x,self.points[zi])

        return (nk,sum_x)


    def printI(self):
        print "N:"+str(self.N)
        print "K:"+str(self.K)
        print "z:\n"+str(self.z)
        #print "Phi:\n"+str(self.Phi)

    def getResults(self):
        return (self.iterList,self.Cluster_num_List,self.D_K_alpha_List,self.D_M1_List,self.D_M2_List)


def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"





if __name__=="__main__":
    parser = argparse.ArgumentParser("Dirichlet Process Mixture Model Cluster Inference routine.")
    parser.add_argument('-i','--in_file',help='Path for points file',required=False)
    parser.add_argument('-o','--out_file',help='Path for output estimators file',required=False)
    args=parser.parse_args()
    if not args.in_file:
        args.in_file="gClusters.txt"
    if not args.out_file:
        args.out_file="estimators.txt"
    points=np.loadtxt(args.in_file)
    #print points
    gibbsOb=gibbs(points)
    gibbsOb.printI()
    gibbsOb.sampling(100,0.1)
    iterList,clusterList,D_k_alpha_List,DM1List,DM2List=gibbsOb.getResults()

    #calculate estimator expectation by sampling
    t=10
    D_k_alpha_sample=random.sample(D_k_alpha_List,t)
    D_k_alpha_mean=np.mean(D_k_alpha_sample)
    DM1sample=random.sample(DM1List,t)
    DM1mean=np.mean(DM1sample)
    DM2sample=random.sample(DM2List,t)
    DM2mean=np.mean(DM2sample)
    with open(args.out_file,'w') as of:
        of.write(str(D_k_alpha_mean)+"\n"+str(DM1mean)+"\n"+str(DM2mean))
    print "D_k_alpha_mean:"+str(D_k_alpha_mean)
    print "DM1mean:"+str(DM1mean)
    print "DM2mean:"+str(DM2mean)

    fig=plt.figure()
    #labelSubplot=fig.add_subplot(111)
    ax=fig.add_subplot(411)
    ax.plot(iterList,clusterList,marker='o')
    ax.set_ylabel('Cluster Number')

    bx=fig.add_subplot(412)
    bx.plot(iterList,D_k_alpha_List,marker='o')
    bx.set_ylabel('D(K;alpha)')

    cx=fig.add_subplot(413)
    cx.plot(iterList,DM1List,marker='o')
    cx.set_ylabel('M1-Distance')

    dx=fig.add_subplot(414)
    dx.plot(iterList,DM2List,marker='o')
    dx.set_ylabel('M2-Distance')

    #labelSubplot.set_xlabel('Iteration count')
    plt.show()
