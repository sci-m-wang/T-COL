import pandas as pd
from Dataloader import Adult, German, Water, Titanic, Phoneme
import torch
from Encoders.TabularEncoder import TabEncoder
from prettytable import PrettyTable
from sklearn.cluster import KMeans
import warnings
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
table = PrettyTable(["Methods","Proximity","Centrality"])

kmeans = KMeans(1)

encoder = TabEncoder(adult,adult_data.categoric)

X = encoder.encode(X)

protos = encoder.encode(adult[adult[c] == 1].iloc[:,:-1])
kmeans.fit(protos)
centers = kmeans.cluster_centers_[0]

samples = encoder.encode(pd.read_csv(root_path + "samples.csv",header=0).iloc[:,:-1])
a = encoder.encode(pd.read_csv(root_path + "colt_good_fcs.csv",header=0).iloc[:,:-1])
b = encoder.encode(pd.read_csv(root_path + "colt_near_ncs.csv",header=0).iloc[:,:-1])
c = encoder.encode(pd.read_csv(root_path + "colt_rep_rss.csv",header=0).iloc[:,:-1])
d = encoder.encode(pd.read_csv(root_path + "colt_cos_rss.csv",header=0).iloc[:,:-1])
e = encoder.encode(pd.read_csv(root_path + "colt_cen_rss.csv",header=0).iloc[:,:-1])

d_k = encoder.encode(pd.read_csv(root_path + "dice_k.csv",header=0).iloc[:,:-1])
d_r = encoder.encode(pd.read_csv(root_path + "dice_r.csv",header=0).iloc[:,:-1])
d_g = encoder.encode(pd.read_csv(root_path + "dice_g.csv",header=0).iloc[:,:-1])

def proximity(s,cf):
    s = torch.from_numpy(s)
    cf = torch.from_numpy(cf)
    p = 0
    for each in s:
        d = 1000000000000
        for item in cf:
            dis = torch.sqrt(torch.sum(torch.square(each-item)))
            if dis <= d:
                d = dis
                pass
            pass
        p += d
        pass
    return p/len(s)

def centrality(c,cf):
    c = torch.from_numpy(c)
    cf = torch.from_numpy(cf)
    d = 1000000000000
    for item in cf:
        dis = torch.sqrt(torch.sum(torch.square(c - item)))
        if dis <= d:
            d = dis
            pass
        pass
    return d

pro_a = proximity(samples,a)
pro_b = proximity(samples,b)
pro_c = proximity(samples,c)
pro_d = proximity(samples,d)
pro_e = proximity(samples,e)
pro_dk = proximity(samples,d_k)
pro_dr = proximity(samples,d_r)
pro_dg = proximity(samples,d_r)

cen_a = centrality(centers,a)
cen_b = centrality(centers,b)
cen_c = centrality(centers,c)
cen_d = centrality(centers,d)
cen_e = centrality(centers,e)
cen_dk = centrality(centers,d_k)
cen_dr = centrality(centers,d_r)
cen_dg = centrality(centers,d_g)

table.add_row(["a",pro_a,cen_a])
table.add_row(["b",pro_b,cen_b])
table.add_row(["c",pro_c,cen_c])
table.add_row(["d",pro_d,cen_d])
table.add_row(["e",pro_e,cen_e])
table.add_row(["dk",pro_dk,cen_dk])
table.add_row(["dr",pro_dr,cen_dr])
table.add_row(["dg",pro_dg,cen_dg])

if args.data in ["adult","water","titanic"]:
    spark = encoder.encode(pd.read_csv(root_path + "Spark.csv",header=0).iloc[:,:-1])
    pro_spark = proximity(samples,spark)
    cen_spark = centrality(centers,spark)
    table.add_row(["spark",pro_spark,cen_spark])
    pass

print(table)