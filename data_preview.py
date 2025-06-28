import pandas as pd
import numpy as np
from lowD import lowD
import string
import re

'''
重要数据：
its_cmp['Sports']    用户的兴趣基于glove建模的向量经过低维嵌入压缩到了4维，np.array()格式

cat_cmp['Books'  'Electronics']     话题偏好单个偏好，同上进行了向量化，6维，格式同上

area_cmp['Suburban']   地区同上，转为2维

user_vecs  原本13维，经过以上三个向量嵌入步骤变为实空间R^22的变量

'''

df=pd.read_csv('./data/user_personalized_features.csv')

def getv(i):
    d=df[df['User_ID']=='#21']
    c=df.columns
    return d[c[i]].values[0]
def w2v(file_path):
    embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
file_path='./w2v/glove.twitter.27B.25d.txt'
vec=w2v(file_path)

def lowercase(s):
    s1=""
    for w in s:
        if(w>='A' and w<='Z'):
            s1+=chr(ord(w)+32)
        else:
            s1+=w
    return s1


#个人兴趣的语义向量化
its={}
its_ar=[]#兴趣列表，文字表示
for i in range(1000):
    interests=df['Interests'][i]#兴趣的文字
    its[interests]=0#加入哈希表，出现过
for i in its.keys():
    its_ar.append(i)
its_vecs=[vec[lowercase(i)] for i in its_ar]#出现过的所有兴趣的glove词嵌入表示
#基于欧式距离的低维嵌入
its_low=lowD(its_vecs,4)#一共5个类，线性无关，于是可以无损压缩到5维,但是压缩完第5个维度全0，发现实际只需要4维
its_cmp={}
for i,i_item in enumerate(its.keys()):
    its_cmp[i_item]=its_low[:,i]
'''
测试用例
def d(a,b):
    return np.dot(a-b,a-b)
x=vec['sports']
y=vec['fashion']
print(d(x,y))
print( d(its_cmp['Sports'],its_cmp['Fashion']) )

'''

#默认话题偏好两个话题重要性等价
cates={}
cates_ar=[]
for i in range(1000):
    topics=df['Product_Category_Preference'][i].split(' ')
    for topic in topics:
        if(topic=='&'): continue
        else:
            cates[topic]=1
for t in cates.keys():
    cates_ar.append(t)
cates_vecs=[vec[lowercase(t)] for t in cates_ar]
#基于欧式距离的低维嵌入
cat_low=lowD(cates_vecs,6)#一共7个类，线性无关，于是可以无损压缩到7维，然而最后一位是全0，6维
cat_cmp={}
for i,i_item in enumerate(cates.keys()):
    cat_cmp[i_item]=cat_low[:,i]
'''
测试用例
a=cat_cmp['Books']
b=cat_cmp['Electronics']
def d(x,y):
    return np.dot(x-y,x-y)
d(a,b)
#np.float64(14.30226421356203)
a1=vec['books']
b1=vec['electronics']
d(a1,b1)
#np.float32(14.302264)
'''

area_vec=lowD([vec['urban'],vec['rural'],vec['suburban']],2)
area_cmp={}
area_cmp['Urban']=area_vec[:,0]
area_cmp['Rural']=area_vec[:,1]
area_cmp['Suburban']=area_vec[:,2]

user_vecs=np.zeros((1000,22),dtype=np.float32)
def generate_user_vecs(i):
    bias=-2
    #j是属性索引，j_real是属性在user_vecs的第一列索引,第一个属性j=2,填在0列j_real=0=j+bias，初始bias=-2
    #由于扩展了一些维度，比如地区第4属性占据2维（23列），后面的5在user_vecs的索引只能从5开始，bias更新为0
    for j in range(2,15):#4地区6兴趣11话题偏好
        j_real=j+bias
        if(j==4):
            user_vecs[i,2:4]=area_cmp[ df[df.columns[4]][i] ]
            bias=-1
        elif(j==6):
            user_vecs[i,5:9]=its_cmp[ df[df.columns[6]][i] ]
            bias=2
        elif(j==11):
            cat_preference=df[df.columns[11]][i].split(' ')
            if(len(cat_preference)>1): del cat_preference[1]
            n=len(cates)
            cat_total=np.zeros(6)
            for cat in cat_preference:
                cat_total+=cat_cmp[cat]
            cat_total/=7
            user_vecs[i,13:19]=cat_total
            bias=7
        elif(j==3):
            user_vecs[i,j_real]=1 if df[df.columns[j]][i]=='Male' else -1
        elif(j==14):
            user_vecs[i,j_real]=1 if df[df.columns[j]][i] else -1
        else:
            user_vecs[i,j_real]=df[df.columns[j]][i]

for i in range(1000):
    generate_user_vecs(i)
i_list=[0,1,4,9,10,11,12,19,20,21]#不涉及向量距离量化语义差距的部分，归一化处理
#事实上，语义向量部分基本上标准差和1同一数量级，均值均为0，因而也不需要特殊处理
#特殊处理也有思路的：Xn是n维序列，几何中心X~方差为var=||（Xn-X~）||^2(L2范数)/num,开根号得std
#令Xn为（Xn-X~）/std，Xn视作一个变量的一个维度，梯度下降应该是可以进行的
for i in i_list:
    user_vecs[:,i]=(user_vecs[:,i]-user_vecs[:,i].mean())/user_vecs[:,i].std()
x=user_vecs
np.save('save_x',x) 



    
  
    
    



