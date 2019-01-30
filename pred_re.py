import os
import glob
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants

def add_jpeg_decoding(h, w, d):
  input_height, input_width, input_depth = (h, w, d)
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  # Convert from full range of uint8 to range [0,1] of float32.
  decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                        tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  return jpeg_data, resized_image

def read_categories(dir):
    ans = []
    f = open(dir)
    line = f.readline()
    while line:
        cate = line.split('.')
        ans.append(cate[1][:-1])
        line = f.readline()
    f.close()
    return ans

path = 'model'
categories = read_categories('categories.txt')
img_size = 299
num_channels = 3

jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(img_size, img_size, num_channels)

img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
pred_dir = os.path.join('pred', '*g')
img_fls = glob.glob(pred_dir)
imgs = []
sess = tf.Session()
for img_fl in img_fls:
    image_data = tf.gfile.FastGFile(img_fl, 'rb').read()
    resized_input_values = sess.run(decoded_image_tensor,
                                  {jpeg_data_tensor: image_data})
    imgs.append(resized_input_values)

tf.saved_model.loader.load(sess, [tag_constants.SERVING], path)
x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
z = tf.get_default_graph().get_tensor_by_name('final_result:0')

pred = tf.nn.softmax(z)
correct = 0
max_size = 5
for img in imgs:
    res = sess.run(pred, {x: img})[0]
    max = 0
    max_i = 0
    for i in range(len(res)):
        if res[i] > max:
            max = res[i]
            max_i = i
    if max_i == 55:
        correct += 1
    res_np = np.array(res)
    n_largest_i = res_np.argsort()[-max_size:][::-1]
    n_largest_cat = []
    for index in n_largest_i:
        n_largest_cat.append([categories[index], res[index]])
    print(n_largest_cat)
    
print('total: ' + str(len(imgs)) + ', correct: ' + str(correct))