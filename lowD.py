import numpy as np
import math

#正交化
def schmidt(vec):
    #因为是近似特征值，无法判断同一特征值，修改一下施密特正交使得不用
    #区分特征值也可以生效
    m=len(vec)
    for i in range(m):
        p=vec[:,i]
        p1=vec[:,i]
        for j in range(i):
            q=vec[:,j]
            p1=p1-(np.dot(p,q)/np.dot(q,q))*q
        if(np.dot(p1,p1)>0): p1=p1/(math.sqrt(np.dot(p1,p1)))
        vec[:,i]=p1
    return vec


#基于欧式距离的低维嵌入
def a(b,c,d):
    return np.dot(b,np.dot(c,d))

def reconstruct(dist2,d):#已知欧式度量矩阵dist2=mat(i,j距离平方)，把原来的高维变量x降维到d
    m=len(dist2)
    sr=np.sum(dist2,axis=1).reshape(m,1) #Sum(D,r行)
    sc=sr.T
    s=np.sum(sr)# Sum(D)
    B=np.zeros((m,m))#计算参考周志华第十章，利用广播机制
    B=(sr/m+sc/m-s/(m*m)-dist2)/2
    vals,vecs=np.linalg.eig(B)
    pairs=[]
    lambd=lambd1=np.zeros((m,m))#对角阵
    P=np.zeros((m,m))#特征向量阵
    for i in range(m):
        pairs.append([vals[i],vecs[:,i]] )
    psort=sorted(pairs,key=lambda x:x[0],reverse=True)
    for i,pair in enumerate(psort):
        lambd1[i,i]=(pair[0])
        if(i<d):
            if(pair[0]>=0): lambd[i,i]=math.sqrt(pair[0])
            #说明本来此位置非负但由于解特征值采用数值算法得到特征0的近似肯能为小负数
            P[:,i]=pair[1]
        else:
            P[:,i]=pair[1]
    P=schmidt(P)#正交化规范化
    Z=np.dot(lambd[0:d,:],P.T)
    return Z
        
def lowD(x,d):#由于项目的特殊性，x是列表，元素是array也是高维变量本身
    m=len(x)
    dist2=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            dist2[i,j]=np.dot(x[i]-x[j],x[i]-x[j])
    return reconstruct(dist2,d)


#测试用例
#x=lowD([np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])] ,2)
    
    
