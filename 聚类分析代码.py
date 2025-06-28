from algorithm_user_image import AP_cluster,analyse,Kmeans
import pandas as pd
import numpy as np

df=pd.read_csv('./data/user_personalized_features.csv')
user_vecs=np.load('save_x.npy')


'''

#AP算法
apc=AP_cluster(user_vecs)
centers,tags=apc.AP_algorithm()#聚类得出了中心和每个样本的类

anal=analyse(tags,df)

for i in range(centers.size):
    temp=anal.get(i,10)
    print("第%d个聚类中心有%d个样本，均价值%f"%(i,temp[0],temp[1]))

'''


'''肘图选K  本例中15最佳
KM=Kmeans(user_vecs)
KM.elble_draw(20)
'''


'''Kmeans聚类指定K为15
KM=Kmeans(user_vecs)
centers,tags,inertia=KM.Kmeans_plus(15)

anal=analyse(tags,df)

for i in range(len(centers)):
    temp=anal.get(i,10)
    print("第%d个聚类中心有%d个样本，均价值%f"%(i,temp[0],temp[1]))

'''

KM=Kmeans(user_vecs)
centers,tags,inertia=KM.Kmeans_plus(15)

anal=analyse(tags,df)

vals=np.zeros(len(centers))  
for i in range(len(centers)):
    temp=anal.get(i,10)
    print("第%d个聚类中心有%d个样本，均价值%f"%(i,temp[0],temp[1]))
    vals[i]= temp[1] 
np.save('centers',centers)
np.save('tags',tags)
np.save('vals',vals)



