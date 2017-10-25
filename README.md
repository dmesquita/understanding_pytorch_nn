# Classifying Text with Neural Networks and Pytorch
Almost any beginner example of Pytorch is about image classification, so I wrote this one about how to classify texts :)

And if you want to lean more about Neural Networks you can check  [this blog post](https://medium.freecodecamp.org/big-picture-machine-learning-classifying-text-with-neural-networks-and-tensorflow-d94036ac2274)

***

```python
import torch
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
```


```python
x = torch.IntTensor([1,3,6])
y = torch.IntTensor([1,1,1])
result = x + y
print(result)
```


     2
     4
     7
    [torch.IntTensor of size 3]




```python
categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print('total texts in train:',len(newsgroups_train.data))
print('total texts in test:',len(newsgroups_test.data))
```

    total texts in train: 1774
    total texts in test: 1180



```python
vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1

for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1

total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i

    return word2index

word2index = get_word_2_index(vocab)
```


```python
def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size:i*batch_size+batch_size]
    categories = df.target[i*batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1

        batches.append(layer)

    for category in categories:
        index_y = -1
        if category == 0:
            index_y = 0
        elif category == 1:
            index_y = 1
        else:
            index_y = 2
        results.append(index_y)


    return np.array(batches),np.array(results)
```


```python
# Parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
hidden_size = 100      # 1st layer and 2nd layer number of features
input_size = total_words # Words in vocab
num_classes = 3         # Categories: graphics, sci.space and baseball
```


```python
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
```


```python
class OurNet(nn.Module):
     def __init__(self, input_size, hidden_size, num_classes):
        super(OurNet, self).__init__()
        self.layer_1 = nn.Linear(input_size,hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_layer = nn.Linear(hidden_size, num_classes, bias=True)

     def forward(self, x):
        out = self.layer_1(x)
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out
```


```python
# input [batch_size, n_labels]
# output [max index for each item in batch, ... ,batch_size-1]
loss = nn.CrossEntropyLoss()
input = Variable(torch.randn(2, 5), requires_grad=True)
print(">>> batch of size 2 and 5 possible classes")
print(input)
target = Variable(torch.LongTensor(2).random_(5))
print(">>> array of size 'batch_size' with the index of the maxium label for each item")
print(target)
output = loss(input, target)
output.backward()
```

    >>> batch of size 2 and 5 possible classes
    Variable containing:
     0.3048 -0.3044  1.1260 -1.0208 -0.1514
     0.0144  1.1776  0.9862  1.2988  0.2670
    [torch.FloatTensor of size 2x5]

    >>> array of size 'batch_size' with the index of the maxium label for each item
    Variable containing:
     3
     0
    [torch.LongTensor of size 2]




```python
net = OurNet(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

# Train the Model
for epoch in range(num_epochs):
    total_batch = int(len(newsgroups_train.data)/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x,batch_y = get_batch(newsgroups_train,i,batch_size)
        articles = Variable(torch.FloatTensor(batch_x))
        labels = Variable(torch.LongTensor(batch_y))
        #print("articles",articles)
        #print(batch_x, labels)
        #print("size labels",labels.size())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(articles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 4 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(newsgroups_train.data)//batch_size, loss.data[0]))
```

    Epoch [1/10], Step [4/11], Loss: 0.7982
    Epoch [1/10], Step [8/11], Loss: 0.4047
    Epoch [2/10], Step [4/11], Loss: 0.0213
    Epoch [2/10], Step [8/11], Loss: 0.0048
    Epoch [3/10], Step [4/11], Loss: 0.1045
    Epoch [3/10], Step [8/11], Loss: 0.3725
    Epoch [4/10], Step [4/11], Loss: 0.0000
    Epoch [4/10], Step [8/11], Loss: 0.0000
    Epoch [5/10], Step [4/11], Loss: 0.0000
    Epoch [5/10], Step [8/11], Loss: 0.0000
    Epoch [6/10], Step [4/11], Loss: 0.0000
    Epoch [6/10], Step [8/11], Loss: 0.0000
    Epoch [7/10], Step [4/11], Loss: 0.0000
    Epoch [7/10], Step [8/11], Loss: 0.0000
    Epoch [8/10], Step [4/11], Loss: 0.0011
    Epoch [8/10], Step [8/11], Loss: 0.0000
    Epoch [9/10], Step [4/11], Loss: 0.0371
    Epoch [9/10], Step [8/11], Loss: 0.0005
    Epoch [10/10], Step [4/11], Loss: 0.0001
    Epoch [10/10], Step [8/11], Loss: 0.1700



```python
# Test the Model
correct = 0
total = 0
total_test_data = len(newsgroups_test.target)
batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
articles = Variable(torch.FloatTensor(batch_x_test))
labels = torch.LongTensor(batch_y_test)
outputs = net(articles)
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum()

print('Accuracy of the network on the 1180 test articles: %d %%' % (100 * correct / total))
```

    Accuracy of the network on the 1180 test articles: 91 %
