#Simple Regression example to choose better m and b values
#for the equation y = mx + b

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)
plt.show()

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)
plt.plot(x_data,y_label,'*')

np.random.rand(2)
m = tf.Variable(.442)
b = tf.Variable(.877)

error = 0
for x,y in zip(x_data,y_label):
    y_hat = m*x + b
    
    #error will be pretty bad at first
    error += (y-y_hat)**2
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    
    #adjust as you see fitting
    training_steps = 10
    
    for i in range(training_steps):
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m,b])
    
x_test = np.linspace(-1,11,10)
#y = mx+b
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'b')
plt.plot(x_data,y_label,'*')    
    
