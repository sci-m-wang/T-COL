from Dataloader import Adult, German, Water, Titanic, Phoneme
import pandas as pd
import torch
import numpy as np
from CreateCFsWithGLT import Creator
from Encoders.TabularEncoder import TabEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import *
from sklearn.metrics import f1_score
from random import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump
from vali_model import Net
import dice_ml

import argparse

## 设定参数，包含原型样本选取规则、局部特征筛选规则、数据集、局部贪心树深度、gpu设备、验证模型、目标类别及样本类别
parse = argparse.ArgumentParser()
parse.add_argument("-p", "--proto", choices=["near","rep","cen", "cos", "good"], default="rep",
                   help="Choosing a selection methods of prototype sample(s) which are used to lead the counterfactual generation.\
                    You can separately 'near', 'rep'& 'cent'. 'near' means that selecting the nearest sample as the prototype sample,\
                    'rep' means that selecting the sample which have the highest possibility as the most representative sample,\
                    'cen' means that selecting the sample located in the center,\
                    'ncos' means that selecting the sample which have the highest cosine similarity.")
parse.add_argument("-f", "--func", choices=["fcs","ncs","rss"], default="rss",
                   help="Choosing a criteria to evaluate local feature combination.\
                   You can select 'lrs','pxm', 'lrs' indicates local relative similarity, 'pxm' means proximity.")
parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "phoneme"], default="adult", help="Choosing a dataset.")
parse.add_argument("-dp", "--depth", required=True, type=int, help="Setting the depth of local greedy tree.")
parse.add_argument("-g", "--gpu", required=True, type=int, help="Selecting gpu number.")
parse.add_argument("-vm", "--vali_model", choices=["RF", "NB", "DT", "SVM", "MLP", "KNN"], default="RF")
parse.add_argument("-dl", "--d_label", required=True, type=int, help="Desired label.")
parse.add_argument("-sl", "--s_label", required=True, type=int, help="Label of samples.")
parse.add_argument("-n", "--n_ces", type=int, default=3, help="Numbers of CEs.")
args = parse.parse_args()

## 定义模型，选取验证模型
RF = RandomForestClassifier()
NB = GaussianNB()
DT = DecisionTreeClassifier()
SVM = SVC()
MLP = MLPClassifier(max_iter=1000)
KNN = KNeighborsClassifier()

# Model initialization.
print("========== validation model initialization... ==========")
v_model = RF
others = []
if args.vali_model == "RF":
    v_model = RF
    others = [NB, DT, SVM, MLP, KNN]
    pass
elif args.vali_model == "NB":
    v_model = NB
    others = [RF, DT, SVM, MLP, KNN]
    pass
elif args.vali_model == "DT":
    v_model = DT
    others = [RF, NB, SVM, MLP, KNN]
    pass
elif args.vali_model == "SVM":
    v_model = SVM
    others = [RF, NB, DT, MLP, KNN]
    pass
elif args.vali_model == "MLP":
    v_model = MLP
    others = [RF, NB, DT, SVM, KNN]
    pass
elif args.vali_model == "KNN":
    v_model = KNN
    others = [RF, NB, DT, SVM, MLP]
    pass
print("========== model created. ==========")

## 确定类别属性
c = ""
root_path = ""

## 导入数据
print("========== data loading... ==========")
global _data,data
if args.data == "german":
    _data = German()
    data = _data.load_data()
    c = "credits"
    root_path = "German/"
    pass
elif args.data == "adult":
    _data = Adult()
    data = _data.load_data()
    c = "income"
    root_path = "Adult/"
    pass
elif args.data == "water":
    _data = Water()
    data = _data.load_data()
    c = "Potability"
    root_path = "Water/"
    pass
elif args.data == "phoneme":
    _data = Phoneme()
    data = _data.load_data()
    c = "class"
    root_path = "Phoneme/"
    pass
elif args.data == "titanic":
    _data = Titanic()
    data = _data.load_data()
    c = "Survived"
    root_path = "Titanic/"
    pass

# root_path += "effi/"
# print(_data.data)

X = _data.data
y = _data.target
encoder = TabEncoder(data,_data.categoric)
X = encoder.encode(X)
print("========== data loaded. ==========")

## 训练模型，设置验证模型
v_model.fit(X, y)
for each in others:
    each.fit(X ,y)
    pass

print("========== verification model initialization... ==========")

dump(v_model,root_path + "v_model.joblib")
model = Net(root_path + "v_model.joblib")

print("========== verification model created. ==========")

## 划分样本
samples = data[data[c] == args.s_label]
protos = data[data[c] == args.d_label]

n_ces = args.n_ces
n_samples = 5

index = randint(0, len(samples)-1-n_samples)
# print(index)
s = samples.iloc[index:index+n_samples]

s.to_csv(root_path + "samples.csv",index=False)

creator = Creator(model, data, s, _data.categorical_features, args.depth, args.d_label, n_ces)

# a
args.func = "fcs"
args.proto = "good"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)            # 默认german,cos,fcs a
CEs.to_csv(root_path + "colt_good_fcs.csv",index=False)

# b
args.func = "ncs"
args.proto = "near"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)
CEs.to_csv(root_path + "colt_near_ncs.csv",index=False)

# c
args.func = "rss"
args.proto = "rep"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)
CEs.to_csv(root_path + "colt_rep_rss.csv",index=False)

# d
args.func = "rss"
args.proto = "cos"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)
CEs.to_csv(root_path + "colt_cos_rss.csv",index=False)

# e
args.func = "rss"
args.proto = "cen"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)
CEs.to_csv(root_path + "colt_cen_rss.csv",index=False)
