import tensorflow as tf
#Initialize the session
sess = tf.InteractiveSession()

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
#print the product
print(product.eval())

#close the session to release resources
sess.close()
