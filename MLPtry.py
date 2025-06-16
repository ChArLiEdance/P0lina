import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test_ori=x_test 
# 打印数据集信息
print('训练集图像数据形状：', x_train.shape)
print('训练集标签数据形状：', y_train.shape)
print('测试集图像数据形状：', x_test.shape)
print('测试集标签数据形状：', y_test.shape)

# 绘制前20张训练集图像
plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()


#28*28=784 change shape to 1*784 and change data type to float32 and normalize from 0~255 to 0-1
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
# Convert labels to one-hot encoding

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    epochs=2, 
                    batch_size=128, 
                    validation_data=(x_test, y_test))

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# 预测结果
predictions = model.predict(x_test)

# 预测第10个测试数据的结果
predictions = model.predict(np.array([x_test[9]]))
print("预测结果：", np.argmax(predictions))

# 绘制第10个测试数据的图形
plt.imshow(x_test_ori[9], cmap=plt.cm.binary)
plt.show()