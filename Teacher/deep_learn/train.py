import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape,y_train.shape)
x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]
x_train = tf.cast(tf.expand_dims(x_train, -1), tf.float32)
x_test = tf.cast(tf.expand_dims(x_test, -1), tf.float32)
x_train = x_train / 255
x_test = x_test / 255
# y_train=tf.cast(tf.expand_dims(y_train,-1),tf.float32)
# plt.imshow(x_train[0])
# plt.show()
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     # plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i])
#     plt.xlabel(y_train[i])
# plt.show()
# print(x_train.shape,y_train.shape)
# x_train = np.random.random((1000, 32))
# y_train = np.random.randint(10, size=1000)

# x_val = np.random.random((200, 32))
# y_val = np.random.randint(10, size=200)

# x_test = np.random.random((200, 32))
# y_test = np.random.randint(10, size=200)

# 模型
model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=[28, 28, 1]),
    tf.keras.layers.AveragePooling2D((3, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.AveragePooling2D((3, 3)),
    tf.keras.layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()
# 编译优化器,损失函数，指标
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              # loss='mse',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))
history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test[:1000], y_test[:1000]))
# model.summary()
print(history.history)
# 绘制曲线
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.title('Train and Validation Loss')
plt.plot(epochs, history.history['loss'], 'b', label='Train Loss')
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
plt.legend()
plt.savefig('./checkpoint/loss.jpg')

plt.figure()
plt.title('Train and Validation Accuracy')
plt.plot(epochs, history.history['accuracy'], 'b', label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation Accuracy')
plt.legend()
plt.savefig('./checkpoint/accuracy.jpg')
plt.show()

result = model.evaluate(x_test, y_test)
print('Test loss', result[0])
print('Test accuracy', result[1])

model.save('./checkpoint/model.h5')
