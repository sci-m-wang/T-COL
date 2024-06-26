from zhipuai import ZhipuAI
import argparse
import pandas as pd
import time

## GLM-4

client = ZhipuAI(api_key="")

messages = [
    {"role":"system", "content":"""
# Role：反事实生成器

## Profile
- Language：中文
- Description：精通反事实推理，善于生成关于查询样本的反事实样例。

## Background
分类任务中反事实解释是对于某个样本变为期忘类别所需做出改变的一种解释，可以表现为一种基于当前样本而衍生出的目标类别的数据样本。反事实解释的生成过程可以看作求解目标样本点的过程，最终找到符合要求的样本点。

## Goal
根据用户输入样本生成反事实样本。

## Skills
- 了解各种常见的Tabular数据集，熟悉其中各种属性及类别的含义。
- 擅长反事实推理，能够迅速发现期忘类别的样本。

## Workflow
- 分析用户输入的数据样本的含义，包括特征值及类别。
- 对样本进行反事实推理，递归地调整样本特征值。
  - 判断调整后的样本是否符合目标类别，若符合则保存反事实样本；否则继续调整。
- 输出反事实样本。

## Output Format
仅输出反事实样本的特征值及类别，以","分隔，每个反事实样本之间换行，不要输出推理过程。

## Example
以Adult Income数据集中的一个样本为例：
- 输入：
  样例：29,Government,HS-grad,Single,Service,White,Male,20,0
  属性含义：age,workclass,education,marital_status,occupation,race,gender,hours_per_week,income
  类别：0，表示：income
  反事实样例数：2
- 输出：
  33,Private,Bachelors,Married,Service,White,Male,40,1
  31,Private,Bachelors,Married,Professional,White,Male,65,1

## Constraint
- 不能改变不可改变的特征，例如种族、性别等。
- 不能做出不可行的改变，例如通常情况下年龄只能增加、学历只会变高。
- 不要回复推理过程。

## Initialization
遵循[Workflow]定义的流程，在[Constraint]的约束下生成反事实样本，并按照[Output Format]指定的格式回复。
     """}
]

def generate(sample,cols,target,cl):
    prompt = """
样例：{}
属性含义：{}
类别：{}，表示：{}
反事实样例数：5
""".format(sample,cols,target,cl)
    messages.append({"role":"user", "content":prompt})
    response = client.chat.completions.create(model="glm-4",messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "phoneme"], default="adult", help="Choosing a dataset.")
    args = parse.parse_args()
    if args.data == "german":
        c = "credits"
        root_path = "German/"
        pass
    elif args.data == "adult":
        c = "income"
        root_path = "Adult/"
        pass
    elif args.data == "water":
        c = "Potability"
        root_path = "Water/"
        pass
    elif args.data == "phoneme":
        c = "class"
        root_path = "Phoneme/"
        pass
    elif args.data == "titanic":
        c = "Survived"
        root_path = "Titanic/"
        pass
    CEs = []
    samples = pd.read_csv(root_path+"samples.csv",header=0)
    samples.astype(str)
    cols = ",".join(samples.columns.tolist()[:-1])
    for sample in samples.iterrows():
        t = sample[1][c]
        sample[1].drop(c, inplace=True)
        response = generate(sample[1],cols,t,c)
        CEs.append(response)
        time.sleep(1)
    
    with open(root_path+"GLM.txt",'w',encoding="utf-8") as f:
        f.write("\n".join(map(str,CEs)))
        pass
    f.close()