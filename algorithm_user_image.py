import sklearn
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation as AP

class AP_cluster():
    dataset=None
    S=None#相似度矩阵
    centers=None
    tags=None
    def __init__(self,data_set):
        self.dataset=data_set
    def d(self,i,j):
        delta=self.dataset[i]-self.dataset[j]
        return math.sqrt( np.dot( delta,delta ) )
    def getS(self):#初始化相似度
        n=len(self.dataset) #行数
        self.S=np.zeros((n,n),dtype=np.float32)
        for i in range(n):
            for j in range(n):
                self.S[i,j]=-self.d(i,j)
        return self.S
    def AP_algorithm(self):
        self.getS()
        self.centers,self.tags=sklearn.cluster.affinity_propagation(self.S, preference=None,\
                                                       convergence_iter=15,\
                                                       max_iter=200, damping=0.5)
        return self.centers,self.tags

class Kmeans():
    dataset=None
    centers=None
    tags=None
    def __init__(self,data_set):
        self.dataset=data_set

    def Kmeans_plus(self,K=8):
        KM=sklearn.cluster.KMeans(n_clusters=K,n_init=5,init='k-means++',max_iter=200)
        KM.fit(self.dataset)
        self.centers=KM.cluster_centers_
        self.tags=KM.labels_
        return self.centers,self.tags,KM.inertia_

    def elble_draw(self,K_max):#绘制肘图选择合适K值,K<=K_max
        X=range(1,K_max+1)
        Y=[]
        for K in range(1,K_max+1):
            Y.append(self.Kmeans_plus(K)[2])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        #X，Y为列表，自己建一下
        plt.figure(figsize=(8, 4))
        plt.plot(X, Y, marker='o', linestyle='--')
        #线标记上圆圈o，线性是--
        plt.xlabel('K值')
        plt.ylabel('误差平方和')
        plt.title('手肘法图表')
        plt.savefig('手肘法图选择K.png',dpi=300)
        plt.grid(True)
        plt.show()


class analyse():
    tags=None
    df=None
    def __init__(self,tags1=None,df1=None):
        self.tags=tags1
        self.df=df1
    def get(self,tag_id,i):#查询类id，对应原始数据的第i个属性
        try:
            sum_=0
            num=0
            for _id in range(len(self.tags)):
                if(self.tags[_id]==tag_id):
                    sum_+=self.df[  self.df.columns[i]  ][_id]
                    num=num+1
            return num,sum_/num#返回簇的大小和属性i均值
        except:
            print('错误,该属性无法求取均值')
    def getmost_discrete(self,tag_id,column_val,column_id):
        num=0
        for _id in range(len(self.tags)):
            if(self.tags[_id]==tag_id):
                if( self.df[  self.df.columns[ column_id ]  ][_id] ==column_val  ):
                    num+=1
        return num
    def induce_discrete(self,tag_id,column_ids):#对于簇序号为tag_id的，分析离散属性的分布并展示
        for col in column_ids:
            print('属性%s的分布情况如下'%(self.df.columns[col]))
            col_val_cnt={}
            for _id in range(len(self.tags)):
                if(self.tags[_id]==tag_id):
                    if(self.df[  self.df.columns[ col ]  ][_id] in col_val_cnt):#此元素出现过了
                        col_val_cnt[ self.df[  self.df.columns[ col ]  ][_id] ]+=1
                    else:
                        col_val_cnt[ self.df[  self.df.columns[ col ]  ][_id] ]=1
            sum_=0
            for key,item in col_val_cnt.items():
                sum_+=item
            for key,item in col_val_cnt.items():
                print("值为%s的有%f%%"%(key,100*item/sum_))
            print("\n")
            


            
            
        


    
