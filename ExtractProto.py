import torch
import numpy as np
import pandas as pd
from Encoders.TabularEncoder import TabEncoder
from sklearn.cluster import KMeans

class GetPrototypes():
    def __init__(self,model,datas,cat,n_protos,target, enc):
        self.model = model
        self.datas = datas
        self.cat = cat
        self.n_protos = n_protos
        self.target = target
        self.enc = enc
        pass
    def get_protos(self, way, samples, dataset):
        global protos,indices, c
        if dataset == "adult":
            c = "income"
            pass
        elif dataset == "german":
            c = "credits"
            pass
        elif dataset == "titanic":
            c = "Survived"
            pass
        elif dataset == "water":
            c = "Potability"
            pass
        elif dataset == "airline":
            c = "Flight Status"
            pass
        elif dataset == "phoneme":
            c = "class"
            pass
        datas = self.enc.encode(self.datas[self.datas[c] == self.target].iloc[:, :-1])
        protos = []
        datas = torch.tensor(np.array(datas))           # datas 表示目标类别
        if way == "near":
            dis = []
            for row in range(len(samples)):
                dist = torch.tensor([0]*len(datas))
                each = pd.DataFrame(samples.iloc[row]).T.iloc[:,:-1]
                each = torch.from_numpy(self.enc.encode(each))
                for i,item in enumerate(datas):
                    distinct = torch.sqrt(torch.sum((each-item)**2))
                    dist[i] = distinct
                    pass
                dis.append(dist)
                pass
            indices = []
            for d in dis:
                index = torch.sort(d).indices[0:self.n_protos].data
                indices.append(index)
                indices = list(set(indices))
                pass
            indices = torch.cat(indices, dim=0)
            for i in indices:
                protos.append(self.datas.iloc[i.item()][:-1].tolist())
                pass
            protos = np.array((protos))
            pass
        elif way == "cen":
            indices = []
            kmeans = KMeans(self.n_protos)
            kmeans.fit(datas.numpy())
            centers = kmeans.cluster_centers_
            for each in centers:
                tensors = torch.from_numpy(each)
                d = 10000000000
                index = 0
                for inx,item in zip(range(len(datas)), datas):
                    # if item.equal(tensors):
                    #     indices += inx
                    #     pass
                    dis = torch.sqrt(torch.sum(torch.square(item-tensors)))
                    if dis <= d:
                        index = inx
                        d = dis
                    pass
                indices.append(index)
                pass
            for i in indices:
                protos.append(self.datas.iloc[i][:-1].tolist())
                pass
            protos = np.array((protos))
            indices = torch.tensor(indices)
            pass
        elif way == "cos":
            dis = []
            for row in range(len(samples)):
                dist = torch.tensor([0] * len(datas))
                each = pd.DataFrame(samples.iloc[row]).T.iloc[:, :-1]
                each = torch.from_numpy(self.enc.encode(each)).squeeze()
                for i, item in enumerate(datas):
                    distinct = torch.cosine_similarity(item,each,0)
                    dist[i] = distinct
                    pass
                dis.append(dist)
                pass
            indices = []
            for d in dis:
                index = torch.sort(d).indices[0:self.n_protos].data
                indices.append(index)
                indices = list(set(indices))
                pass
            indices = torch.cat(indices, dim=0)
            for i in indices:
                protos.append(self.datas.iloc[i.item()][:-1].tolist())
                pass
            protos = np.array((protos))
            pass
        elif way == "rep":
            outs = self.model.predict_proba(datas)
            losses = torch.abs(outs[:, self.target] - self.target)
            indices = torch.sort(losses).indices[0:self.n_protos].data
            for i in indices:
                protos.append(self.datas.iloc[i.item()][:-1].tolist())
                pass
            protos = np.array(protos)
            pass
        elif way == "good":
            indices = []
            dis = []
            targets = self.datas[self.datas[c] == self.target].iloc[:,:-1]
            for row in range(len(samples)):
                dist = []
                for tar in range(len(targets)):
                    dist.append(np.sum(samples.iloc[row].tolist() == targets.iloc[tar].tolist()))
                    pass
                dis.append(dist)
                pass
            for d in dis:
                index = torch.sort(torch.tensor(d)).indices[0:self.n_protos].data
                indices.append(index)
                indices = list(set(indices))
                pass
            indices = torch.cat(indices, dim=0)
            for i in indices:
                protos.append(self.datas.iloc[i.item()][:-1].tolist())
                pass
            protos = np.array((protos))
            pass
        return protos,indices
    pass
