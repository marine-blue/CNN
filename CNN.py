import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

datasets = scipy.io.loadmat("./usps.mat")

train_examples = []
test_examples = []
train_examples = datasets['trai'].reshape(1,-1,order='F')
train_examples = train_examples.reshape(4649,16,16)
test_examples = datasets['test'].reshape(1,-1,order = 'F')
test_examples = test_examples.reshape(4649,16,16)

t=np.random.randint(0,100)
train_labels = datasets['trai_label']
test_labels = datasets['test_label']

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((16,16,1),input_shape=(16,16)),
    tf.keras.layers.Conv2D(32,(4,4),activation = 'relu'),
    tf.keras.layers.Conv2D(32,(2,2),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
model.fit(train_examples,train_labels,batch_size=16,epochs=10)
loss, accuracy =model.evaluate(test_examples,test_labels,batch_size=32)
predict = tf.argmax(model.predict(test_examples,),axis = -1).numpy()
confusion_M = tf.math.confusion_matrix(test_labels,predict)
print(confusion_M)
print("Accuracy:"+str(round(accuracy,4)*100)+"%")
