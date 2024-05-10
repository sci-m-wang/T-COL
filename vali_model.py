import torch
import numpy as np
import torch.nn.functional as F
from Dataloader import Climate
from Encoders.TabularEncoder import TabEncoder
from torch.nn import Sequential
from sklearn.neural_network import MLPClassifier
from joblib import dump,load

## 导入数据
# adult = Climate()
# categoric = []
# X = adult.data
# y = adult.target
# adult = adult.load_data()
# # 数据编码
# encoder = TabEncoder(adult,categoric)
# X = encoder.encode(X)
#
# clf = MLPClassifier()
# clf.fit(X,y)
# dump(clf,"clf.joblib")
#
# X = torch.tensor(np.array(X,dtype='float'),dtype=torch.float).cuda()
# y = torch.tensor(np.array(y,dtype='int'),dtype=torch.long).cuda()

class Net(object):
    def __init__(self,model_path:str):
        self.model = load(model_path)
        pass
    def predict(self,X:torch.Tensor):
        self.device = X.device
        self.X = X.cpu().numpy()
        self.prediction = self.model.predict(self.X)
        return torch.tensor(self.prediction,dtype=torch.long).to(self.device)
    def predict_proba(self,X):
        self.device = X.device
        self.X = X.cpu().numpy()
        self.prediction_proba = self.model.predict_proba(self.X)
        return torch.tensor(self.prediction_proba,dtype=torch.float).to(self.device)
    pass

# net = Net("clf.joblib")
# predictions = net.predict(X)
# max__ = torch.max(net.predict_proba(X[7].reshape(1,-1)).squeeze(),0)
#
# print()
# print(torch.sum(predictions == y)/len(X))