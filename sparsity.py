import pandas as pd
from Dataloader import Adult,German,Water,Titanic,Phoneme
import torch
from Encoders.TabularEncoder import TabEncoder
from prettytable import PrettyTable
from sklearn.cluster import KMeans
import warnings
import numpy as np
import argparse

warnings.filterwarnings("ignore")

parse = argparse.ArgumentParser()
parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "phoneme"], default="adult", help="Choosing a dataset.")
args = parse.parse_args()

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
X = adult_data.data
y = adult_data.target
table = PrettyTable(["Methods","Sparsity"])

samples = pd.read_csv(root_path + "samples.csv",header=0).iloc[:,:-1]
a = pd.read_csv(root_path + "colt_good_fcs.csv",header=0).iloc[:,:-1]
b = pd.read_csv(root_path + "colt_near_ncs.csv",header=0).iloc[:,:-1]
cc = pd.read_csv(root_path + "colt_rep_rss.csv",header=0).iloc[:,:-1]
d = pd.read_csv(root_path + "colt_cos_rss.csv",header=0).iloc[:,:-1]
e = pd.read_csv(root_path + "colt_cen_rss.csv",header=0).iloc[:,:-1]

d_k = pd.read_csv(root_path + "dice_k.csv",header=0).iloc[:,:-1]
d_r = pd.read_csv(root_path + "dice_r.csv",header=0).iloc[:,:-1]
if args.data != "airline":
    d_g = pd.read_csv(root_path + "dice_g.csv",header=0).iloc[:,:-1]
    pass
else:
    d_g = d_r = pd.read_csv(root_path + "dice_r.csv",header=0).iloc[:,:-1]

def sparsity(s,cf):
    s = np.array(s).tolist()
    cf = np.array(cf).tolist()
    p = 0
    for each in s:
        d = 0
        for item in cf:
            for i in range(len(each)):
                for j in range(len(item)):
                    if each[i] == item[j]:
                        d += 1
                        pass
                    pass
                pass
            if d >= p:
                p = d
                pass
            pass
        pass
    return p / len(s[0])


spa_a = sparsity(samples,a)
spa_b = sparsity(samples,b)
spa_c = sparsity(samples,cc)
spa_d = sparsity(samples,d)
spa_e = sparsity(samples,e)
spa_dk = sparsity(samples,d_k)
spa_dr = sparsity(samples,d_r)

spa_dg = sparsity(samples,d_g)


table.add_row(["a",spa_a])
table.add_row(["b",spa_b])
table.add_row(["c",spa_c])
table.add_row(["d",spa_d])
table.add_row(["e",spa_e])
table.add_row(["dk",spa_dk])
table.add_row(["dr",spa_dr])
table.add_row(["dg",spa_dg])

if args.data in ["adult","water","titanic"]:
    spark = pd.read_csv(root_path + "Spark.csv",header=0).iloc[:,:-1]
    spa_spark = sparsity(samples,spark)
    table.add_row(["spark",spa_spark])
    pass


print(table)