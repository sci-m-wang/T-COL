import pandas as pd
import numpy as np
from GLT import LinkTree
from tqdm import tqdm
from ExtractProto import GetPrototypes
import torch
from Encoders.TabularEncoder import TabEncoder

class Creator(object):
    def __init__(self,model,datas:pd.DataFrame,samples:pd.DataFrame,categorical_features,deepth,target_label,n_CFs):
        self.model = model
        self.target_label = target_label
        self.goals = datas[datas.iloc[:,-1] == self.target_label]
        self.samples = samples
        self.deepth = deepth
        self.n_CFs = n_CFs
        self.counterfactuals = []
        self.categorical_features = categorical_features
        self.encoder = TabEncoder(datas,self.categorical_features)
        pass
    def createCFs(self, way, dataset, device,func):
        device = torch.cuda.set_device(device)
        Prototype = GetPrototypes(self.model, self.goals, self.categorical_features, self.n_CFs, self.target_label, self.encoder)  # model用于提取原型；goals为原型类别的样本；n_CFS为生成反事实的数量，也即提取的原型数，对应该类中的n_protos；target_label为目标类别，也就是原型样本的类别
        protos,indices = Prototype.get_protos(way, self.samples, dataset)
        # 保存原始数据中的原型
        in_list = indices.data.cpu().numpy().tolist()
        original_protos = self.goals.iloc[in_list]
        protos = pd.DataFrame(protos,columns=self.goals.columns.values[:-1])
        protos = self.encoder.encode(protos)
        samples = self.encoder.encode(self.samples.iloc[:,:-1])
        protos = np.array(protos, dtype='float')
        samples = np.array(samples)
        # 转化为Tensor
        protos = torch.tensor(protos,dtype=torch.float).to(device)
        samples = torch.tensor(samples,dtype=torch.float).to(device)
        count = 0
        for sample in tqdm(samples):
            reckoning = 0
            for proto in protos:
                CF = []
                tree = LinkTree(self.deepth,proto,sample)
                _,counterfactual_path = tree.create_CF(func)
                for i in range(len(counterfactual_path)):
                    if counterfactual_path[i] == 0:
                        CF.append(self.samples.iloc[count, i])
                        pass
                    else:
                        CF.append(original_protos.iloc[reckoning, i])
                        pass
                    pass
                # CF_ec = TabEncoder(pd.DataFrame(np.array(CF+[self.target_label]).reshape(1,-1),columns=self.goals.columns.values),self.categorical_features)
                counterfactual = self.encoder.encode(pd.DataFrame(np.array(CF).reshape(1,-1),columns=self.goals.columns.values[:-1]))
                # print(self.model.predict(counterfactual.reshape(1,-1)))
                # if torch.max(self.model(counterfactual),0)[1] == self.target_label:
                counterfactual = torch.tensor(np.array(counterfactual.tolist(),dtype='float')).to(device)
                if self.model.predict(counterfactual) == self.target_label:
                    self.counterfactuals.append(tuple(CF+[self.target_label]))
                    pass
                # self.counterfactuals.append(tuple(CF+[self.target_label]))
                reckoning += 1
                pass
            count += 1
            pass
        return self.counterfactuals
    pass