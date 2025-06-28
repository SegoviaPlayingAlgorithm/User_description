# User_description
   用户画像项目

   
(你可以不跑预训练程序，跑聚类分析和归纳就可以看到结果，为什么要说这个呢,因为
如果想跑预训练程序，需要glove预训练的词嵌入结果，但是文件太大，发不上来，你要去斯坦福大学网站下载，推特语料库词嵌入25维词嵌入
放入新建文件夹w2v，叫做glove.twitter.27B.25d.txt)    （w2v和项目里的py程序放在同一文件夹内）


对于用户数据，我们需要对几个词构成的维度进行词嵌入dimension=25，再进行降维（低维嵌入技术参考周志华西瓜书10章）
把用户向量化后，聚类算法聚类


我调用了AP和Kmeans两种方式聚类，前者自适应得指定类数K,但是分成太多类了，每个类数据太少也不稳定，效果不佳。
AP：亲和度算法结果
![AP聚类划分结果](https://github.com/user-attachments/assets/730a025f-44f0-4470-bcbc-2e6697ba5033)


Kmeans使用肘图选择合适的K，K=15为最佳


![手肘法图选择K](https://github.com/user-attachments/assets/0cf4533f-3a77-422a-acc6-b9402ecef9c5)
![Kmeans划分K15结果](https://github.com/user-attachments/assets/62a436bf-1887-4f33-86fb-30156e4aaabb)


对每个聚类簇的成员统计一下各种属性，找出该簇的共性，这就是一种用户画像方式
详细的分析结果储存在用户画像.txt中
---------------------------------------------------
簇号==0,价值为2652.327869,属于高价值用户



属性Gender的分布情况如下

值为Male的有55.737705%

值为Female的有44.262295%


属性Location的分布情况如下

值为Suburban的有100.000000%


属性Interests的分布情况如下

值为Sports的有100.000000%


属性Product_Category_Preference的分布情况如下

值为Books的有16.393443%

值为Apparel的有26.229508%

值为Health & Beauty的有18.032787%

值为Home & Kitchen的有14.754098%

值为Electronics的有24.590164%


属性Newsletter_Subscription的分布情况如下

值为True的有50.819672%

值为False的有49.180328%


![用户画像_展示](https://github.com/user-attachments/assets/19752a92-1384-4d0e-8d28-8559788000fd)
