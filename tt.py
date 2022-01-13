from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model_loaded = tf.keras.models.load_model('./my_model.h5')

dir_to_img = './big/IMG_7859.jpeg'
img_width, img_height = int(3024/10), int(4032/10)

img = load_img('dir_to_img')
img_array = img_to_array(img)
img_array = tf.image.resize(img_array, (img_width, img_height))
img_array = img_array / 255.0
img_expended = np.expand_dims(img_array, axis=0)
prediction = round(float(model_loaded.predict(img_expended)))
pred_label = 'small' if prediction == 1 else 'big'
plt.figure()
plt.imshow(img)
plt.title(f'{pred_label} {prediction}')
plt.show()