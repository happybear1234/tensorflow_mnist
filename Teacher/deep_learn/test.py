import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(_,_),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_test=tf.expand_dims(x_test,-1)
x_test=x_test/255
# x_test=np.random.random((200,32))
# y_test=np.random.randint(10,size=200)

model=tf.keras.models.load_model('./checkpoint/model.h5')
plt.figure(figsize=(10,10))
for i in range(25):
    x_input=tf.expand_dims(x_test[i],0)
    result = model.predict(x_input)
    print('预测：',np.argmax(result),'标签：',y_test[i])
    plt.subplot(5,5,i+1)
    x_input=tf.squeeze(x_input,0)
    plt.imshow(x_input)
plt.show()