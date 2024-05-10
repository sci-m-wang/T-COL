from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from Dataloader import Adult,German,Water,Titanic,Phoneme
from Encoders.TabularEncoder import TabEncoder
import numpy as np
import warnings
from Dataloader import Adult,German
import pandas as pd
from prettytable import PrettyTable
import argparse

warnings.filterwarnings("ignore")

parse = argparse.ArgumentParser()
parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "phoneme"], default="adult", help="Choosing a dataset.")
args = parse.parse_args()

## 真实性验证分类器
RF = RandomForestClassifier()
NB = GaussianNB()
DT = DecisionTreeClassifier()
SVM = SVC()
MLP = MLPClassifier(max_iter=1000)
KNN = KNeighborsClassifier()

def eval_df(dataset, ces):
    global X, y
    if dataset == "adult":
        _data = Adult()
        weights = {'NB':0.74, 'DT':0.69, 'SVM':0.74, 'MLP':0.75, 'KNN':0.73}
        pass
    elif dataset == "german":
        _data = German()
        weights = {'NB':0.7, 'DT':0.65, 'SVM':0.69, 'MLP':0.67, 'KNN':0.66}
        pass
    elif dataset == "water":
        _data = Water()
        weights = {'NB':0.52, 'DT':0.58, 'SVM':0.62, 'MLP':0.64, 'KNN':0.62}
        pass
    elif dataset == "titanic":
        _data = Titanic()
        weights = {'NB':0.93, 'DT':0.94, 'SVM':0.93, 'MLP':0.95, 'KNN':0.93}
        pass
    elif dataset == "phoneme":
        _data = Phoneme()
        weights = {'NB':0.75, 'DT':0.83, 'SVM':0.8, 'MLP':0.82, 'KNN':0.84}
        pass
    else:
        _data = None
        weights = {}
        pass
    data = _data.load_data()
    X = _data.data
    y = _data.target
    encoder = TabEncoder(data, _data.categoric)
    X = encoder.encode(X)
    ## 真实性分类器训练
    NB.fit(X, y)
    DT.fit(X, y)
    SVM.fit(X, y)
    MLP.fit(X, y)
    KNN.fit(X, y)

    RF.fit(X,y)

    NB_score = np.sum(NB.predict(ces))/len(ces)
    DT_score = np.sum(DT.predict(ces))/len(ces)
    SVM_score = np.sum(SVM.predict(ces))/len(ces)
    MLP_score = np.sum(MLP.predict(ces))/len(ces)
    KNN_score = np.sum(KNN.predict(ces))/len(ces)

    RF_score = np.sum(RF.predict(ces))/len(ces)

    df = (NB_score*weights['NB']+DT_score*weights['DT']+SVM_score*weights['SVM']+MLP_score*weights['MLP']+KNN_score*weights['KNN'])/(weights['NB']+weights['DT']+weights['SVM']+weights['MLP']+weights['KNN'])
    return df, RF_score

if args.data == "adult":
    adult_data = Adult()
    root_path = "Adult/"
    c = "income"
    pass
elif args.data == "german":
    adult_data = German()
    root_path = "German/"
    c = "credits"
    pass
elif args.data == "water":
    adult_data = Water()
    root_path = "Water/"
    c = "Potability"
    pass
elif args.data == "titanic":
    adult_data = Titanic()
    root_path = "Titanic/"
    c = "Survived"
    pass
elif args.data == "phoneme":
    adult_data = Phoneme()
    root_path = "Phoneme/"
    c = "class"
    pass
else:
    adult_data = None
    root_path = ""
    pass

adult = adult_data.load_data()
table = PrettyTable(["Methods","Datafidelity","Validity"])
encoder = TabEncoder(adult,adult_data.categoric)

a = encoder.encode(pd.read_csv(root_path + "colt_good_fcs.csv",header=0).iloc[:,:-1])
b = encoder.encode(pd.read_csv(root_path + "colt_near_ncs.csv",header=0).iloc[:,:-1])
cc = encoder.encode(pd.read_csv(root_path + "colt_rep_rss.csv",header=0).iloc[:,:-1])
d = encoder.encode(pd.read_csv(root_path + "colt_cos_rss.csv",header=0).iloc[:,:-1])
e = encoder.encode(pd.read_csv(root_path + "colt_cen_rss.csv",header=0).iloc[:,:-1])

d_k = encoder.encode(pd.read_csv(root_path + "dice_k.csv",header=0).iloc[:,:-1])
d_r = encoder.encode(pd.read_csv(root_path + "dice_r.csv",header=0).iloc[:,:-1])
d_g = encoder.encode(pd.read_csv(root_path + "dice_g.csv",header=0).iloc[:,:-1])

df_a, val_a = eval_df(args.data,a)
df_b, val_b = eval_df(args.data,b)
df_c, val_c = eval_df(args.data,cc)
df_d, val_d = eval_df(args.data,d)
df_e, val_e = eval_df(args.data,e)

df_dk, val_dk = eval_df(args.data,d_k)
df_dr, val_dr = eval_df(args.data,d_r)
df_dg, val_dg = eval_df(args.data,d_g)

table.add_row(["a", df_a, val_a])
table.add_row(["b", df_b, val_b])
table.add_row(["c", df_c, val_c])
table.add_row(["d", df_d, val_d])
table.add_row(["e", df_e, val_e])
table.add_row(["dk", df_dk, val_dk])
table.add_row(["dr", df_dr, val_dr])
table.add_row(["dg", df_dg, val_dg])

if args.data in ["adult","water","titanic"]:
    spark = encoder.encode(pd.read_csv(root_path + "Spark.csv",header=0).iloc[:,:-1])
    df_spark, val_spark = eval_df(args.data, spark)
    table.add_row(["spark",df_spark,val_spark])
    pass

print(table)