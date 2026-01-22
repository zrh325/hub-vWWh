"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化
"""
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

## 读取数据
df = pd.read_csv('dataset.csv', header=None, sep='\t')
## 将输入的句子转换成数字，并且记录对应的字典
text_list = df[0].tolist()
label_list = df[1].tolist()

max_len = 40
char_to_index = {'pad': 0}
for text in text_list:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {index: char for char, index in char_to_index.items()}
vocabulary_size = len(char_to_index)

## 将标签转换成数字
label_to_index = {label: idx for idx, label in enumerate(set(label_list))}
index_to_label = {idx: label for idx, label in enumerate(set(label_list))}
numerical_labels = [label_to_index[label] for label in label_list]


## 创建读取数据规则
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_len, char_to_index, vocabulary_size):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.char_to_index = char_to_index
        self.vocabulary_size = vocabulary_size
        self.bow_vectors = self._generate_bow_vectors()

    def __getitem__(self, index):
        return self.bow_vectors[index], self.labels[index]

    def __len__(self):
        return len(self.texts)

    def _generate_bow_vectors(self):
        ## 将所有输入的文本对应的数字转换成字典中对应数字的个数映射
        bow_vectors = []
        for text_str in self.texts:
            text_number = [self.char_to_index.get(text_char, 0) for text_char in text_str[:self.max_len]]
            bow_vector = torch.zeros(self.vocabulary_size)
            for number in text_number:
                if number != 0:
                    bow_vector[number] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)


class MultiClassifier(nn.Module):

    def __init__(self, input_dim, output_dim, *hidden_dims):
        ## 初始化训练层数
        super(MultiClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dims[3], output_dim)

    def forward(self, x):
        ## 手动实现每层计算
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

## 初始化读取规则
text_dataset = TextDataset(text_list, numerical_labels, max_len, char_to_index, vocabulary_size)
## 批量读取数据集
data_loader = DataLoader(text_dataset, batch_size=50, shuffle=True)


## 定义中间隐藏层维度, 输出的维度
hidden_dims = [128, 256, 512, 1024]
output_dim = len(label_to_index)

## 初始化训练模型
model = MultiClassifier(vocabulary_size, output_dim, *hidden_dims)
## 采用内部损失函数、激活函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


## 将数据集分批次共训练N次
epoches = 20
for epoch in range(epoches):
    model.train()
    running_loss = 0.0
    for index, (text_vocabularys, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = model(text_vocabularys)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 50 == 0:
            print(f'批次个数:{index}, 当前损失值：{loss.item()}')
    print(f'当前为第{epoch + 1}/{epoches}轮训练， 当前训练的损失值为{running_loss/len(data_loader):.4f}')


## 识别函数
def classify_text_label(text, model, char_to_index, vocabulary_size, max_len, index_to_label):
    text_vocabulary = torch.zeros(vocabulary_size)
    text_char_list = [char_to_index.get(text_char, 0) for text_char in text[:max_len]]
    for text_char in text_char_list:
        if text_char != 0:
            text_vocabulary[text_char] += 1

    text_vocabulary = text_vocabulary.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(text_vocabulary)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label


classify_text_list = ['查询长沙明天的天气', '帮我播放一首黄昏', '我想看一下去武汉的车票']
for classify_text in classify_text_list:
    predicted_label = classify_text_label(classify_text, model, char_to_index, vocabulary_size, max_len, index_to_label)
    print(f"输入的字符串为{classify_text} ，预测结果为{predicted_label}")