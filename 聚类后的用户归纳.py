from algorithm_user_image import analyse
import pandas as pd
import numpy as np
import sys

df=pd.read_csv('./data/user_personalized_features.csv')
user_vecs=np.load('save_x.npy')
vals=np.load('vals.npy')
centers=np.load('centers.npy')
tags=np.load('tags.npy')

anal=analyse(tags,df)
column_ids=[3,4,6,11,14]
valuation=""#用户评价

    
for i in range(len(centers)):
    if(vals[i]>2650): valuation="高"
    elif(vals[i]>2450): valuation="中"
    else: valuation="低"
    print('---------------------------------------------------')
    print("簇号==%d,价值为%f,属于%s价值用户"%(i,vals[i],valuation))
    anal.induce_discrete(i,column_ids)#簇=5是消费最高的簇
    print("\n\n")
    print('---------------------------------------------------')


