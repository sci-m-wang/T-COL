import numpy as np
import pandas as pd
import os

PATH = os.path.join("Datasets/")

class Adult():
    """
    UCI成年人收入数据集，Dice文章筛选过的版本
    """
    # columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
    #            "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
    adult = pd.read_csv(PATH+"adult_pre.csv",header=0,index_col=0)
    adult_data = pd.DataFrame(adult)
    # adult_data["income"].replace(" <=50K",0,inplace=True)           # 类别编码
    # adult_data["income"].replace(" >50K", 1, inplace=True)
    def __init__(self):
        self.data = self.adult_data.iloc[:,:-1]
        self.target = self.adult_data["income"]
        self.categoric = ["workclass","education","marital_status","occupation","race","gender"]
        self.continues = self.adult_data.columns.difference(self.categoric).drop("income").values.tolist()
        self.categorical_features = self.adult_data[self.categoric]
        self.continuous_features = self.adult_data[self.continues]
        pass
    def load_data(self):
        return self.adult_data
    pass

class German():
    """
    UCI德国信用卡数据集
    """
    columns = ["Estatus","Duration","Chistory","Purpose","Ncredit","Saccount","EmploySince","Insrate","Pstatus","Odebtors",
               "ResidenceSince","Property","Age","Insplans","House","NTBcredit","Job","NMpeople","Telephone","Fworker","credits"]
    german = pd.read_csv(PATH+"german.csv",names=columns)
    german_data = pd.DataFrame(german)
    german_data["credits"].replace(2,0,inplace=True)
    def __init__(self):
        self.data = self.german_data.iloc[:,:-1]
        self.target = self.german_data["credits"]
        self.categoric = ["Estatus","Chistory","Purpose","Saccount","EmploySince","Pstatus","Odebtors",
                            "Property","Insplans","House","Job","Telephone","Fworker"]
        self.continues = self.german_data.columns.difference(self.categoric).drop("credits").values.tolist()
        self.categorical_features = self.german_data[self.categoric]
        self.continuous_features = self.german_data[self.continues]
        pass
    def load_data(self):
        return self.german_data
    pass

class Water():
    '''
    Kaggle Water质量数据集
    '''
    water = pd.read_csv(PATH+"water_potability.csv",header=0)
    water_data = pd.DataFrame(water)
    water_data.dropna(inplace=True)
    def __init__(self):
        self.data = self.water_data.iloc[:,:-1]
        self.target = self.water_data["Potability"]
        self.categoric = []
        self.continues = self.water_data.columns.drop("Potability").values.tolist()
        self.categorical_features = self.water_data[self.categoric]
        self.continuous_features = self.water_data[self.continues]
        pass
    def load_data(self):
        return self.water_data
    pass

class Titanic():
    '''
    Kaggle Titanic 数据集
    '''
    titanic = pd.read_csv(PATH+"titanic.csv",header=0,index_col=0)
    titanic_data = pd.DataFrame(titanic)
    titanic_data.dropna(inplace=True)
    cols = titanic_data.columns.tolist()
    cols.remove("Survived")
    cols.append("Survived")
    titanic_data = titanic_data.reindex(columns = cols)
    def __init__(self):
        self.data = self.titanic_data.iloc[:,:-1]
        self.target = self.titanic_data["Survived"]
        self.categoric = ["Pclass","Sex","Ticket","Cabin","Embarked"]
        self.continues = self.titanic_data.columns.difference(self.categoric).drop("Survived").values.tolist()
        self.categorical_features = self.titanic_data[self.categoric]
        self.continuous_features = self.titanic_data[self.continues]
        pass
    def load_data(self):
        return self.titanic_data

class Phoneme():
    phoneme = pd.read_csv(PATH + "php8Mz7BG.arff",header=None,
                 sep=",", usecols=range(6),skiprows=10,
                 names = ["V1","V2","V3","V4","V5","class"])
    phoneme_data = pd.DataFrame(phoneme)
    phoneme_data.replace(1,0,inplace=True)
    phoneme_data.replace(2,1,inplace=True)
    def __init__(self):
        self.data = self.phoneme_data.iloc[:, :-1]
        self.target = self.phoneme_data["class"]
        self.categoric = []
        self.continues = self.phoneme_data.columns.difference(self.categoric).drop("class").values.tolist()
        self.categorical_features = self.phoneme_data[self.categoric]
        self.continuous_features = self.phoneme_data[self.continues]
        pass

    def load_data(self):
        return self.phoneme_data

