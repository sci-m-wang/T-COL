import json
import torch
import numpy as np
import torch.nn.functional as F

PATH = "/datas/wangm/GLT-fastCFs/"

tree_path = {}

with open(PATH+"tree_path.json","r") as f:
    tree_path = json.load(f)
    pass

class Tree(object):
    """
    构建局部特征表示树，利用原型及样本的特征构建反事实解释的待填充特征
    """
    deep_trans = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine"}
    def __init__(self,deepth:int,proto:torch.Tensor,sample:torch.Tensor, func:str):
        self.paths = tree_path[self.deep_trans[deepth]]
        self.tree = []
        self.local_simi = []
        # 树生成
        for path in self.paths:
            ptree = torch.tensor([]).to(proto.device)  # 记录当前路径上的局部反事实编码
            for i in range(0,len(path)):
                if path[i] == '1':
                    ptree = torch.cat((ptree,proto[i].reshape(1,)),dim=0)           # 如果路径为1，将原型的对应特征添加进树
                    pass
                else:
                    ptree = torch.cat((ptree,sample[i].reshape(1,)),dim=0)          # 如果路径为0，将样本的对应特征添加进树
                    pass
                pass
            cos = torch.exp(F.cosine_similarity(proto,ptree,0,1e-8))
            # cos = torch.exp(torch.sqrt(torch.sum((proto-ptree)**2)))
            dis = torch.sigmoid(torch.sqrt(torch.sum((sample-ptree)**2)))
            if func == "rss":
                self.local_simi.append(cos/dis)
                pass
            elif func == "ncs":
                self.local_simi.append(F.cosine_similarity(proto,ptree,0,1e-8)/torch.exp(torch.sqrt(torch.sum((sample-ptree)**2))))
                pass
            elif func == "fcs":
                c = torch.sigmoid(F.cosine_similarity(proto,ptree,0,1e-8))
                d = torch.sum(sample==ptree)/len(sample)
                # print(c,d)
                self.local_simi.append(c + d)
            self.tree.append(ptree)
            pass
        pass
    # 产生局部最优解
    def gen_opt(self):
        index = torch.tensor(self.local_simi).sort(descending=True).indices[0].item()
        opt_path = self.paths[index]
        opt = self.tree[index]
        max_local_simi = self.local_simi[index].item()
        return list(map(int,opt_path)),opt,max_local_simi
    pass

class LinkTree(object):
    """
    利用局部树的结果生成反事实解释
    """
    def __init__(self,tree_deepth:int,proto:torch.Tensor,instance:torch.Tensor):
        self.deepth = tree_deepth
        self.proto = proto
        self.instance = instance
        self.optCF = torch.tensor([],device=proto.device)
        self.opt_path = []
        pass
    def create_CF(self,func):
        deep = len(self.proto)
        CF = torch.tensor([],device=self.proto.device)
        CF_path = []
        while deep > 0:
            if deep < self.deepth:
                self.deepth = deep
                pass
            p = self.proto[0:self.deepth]
            self.proto = self.proto[self.deepth:]
            i = self.instance[0:self.deepth]
            self.instance = self.instance[self.deepth:]
            tree = Tree(self.deepth,p,i,func)
            opt_path,opt,_ = tree.gen_opt()
            CF = torch.cat((CF,opt),dim=0)
            CF_path += opt_path         # 为了通用性，这里返回树的路径，然后利用路径以及原型样本生成反事实
            deep -= self.deepth
            pass
        return CF,CF_path
    pass