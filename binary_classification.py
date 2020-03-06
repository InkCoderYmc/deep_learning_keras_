#二分类问题
'''
最广泛的机器学习问题
'''


#数据集
'''
IMDB数据集（互联网电影数据库）
50000条两极分化的评论
'''
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words = 10000
) #参数控制只保留训练数据中前10000个最常出现的单词，舍弃低频单词。

'''
#将评论解码为英文单词
word_index = imdb.get_word_index() #将单词映射为整数索引的字典
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.item()]
) #颠倒键值，将整数索引为单词
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
) #将评论解码，减去3是因为前3个是保留索引
'''

 #手动转换成one-hot向量
import numpy as np

def vectorize_swquences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #创建零矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. #索引设定为1
    return results

#数据向量化
x_train = vectorize_swquences(train_data)
x_test = vectorize_swquences(test_data)

#标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#留出训练集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#构建网络
from keras import models, layers


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#编译模型
model.compile(
    optimizer = 'rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)


'''
#先进行训练测试判断合适的迭代次数
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)    
)

#根据history绘制图像
import matplotlib.pyplot as plt

#绘制训练损失和验证损失
#先根据history获取数据
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

#绘制图像
plt.plot(epochs, loss_values, 'bo', label='Training loss') #蓝色圆点绘制
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') #蓝色实线绘制
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

#绘制训练精度和验证精度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)

print(model.predict(x_test))