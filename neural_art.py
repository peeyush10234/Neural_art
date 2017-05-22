import os
import time
import sys
import numpy as np

from scipy.misc import imsave
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K


"""
Inputs: 
	base_image_path: Content Image Path
	style_reference_image_path: Style Image Path
	iterations: Number of Iterations
	directory: Output Directory
"""
base_image_path = sys.argv[1]
style_reference_image_path = sys.argv[2]
iterations = int(sys.argv[5])
directory = sys.argv[3]

result_prefix = 'image'

# Parameters for style and content weight for generating combined image.
total_variation_weight = 1.0
style_weight = 0.8
content_weight = 0.125

# Dimensions of input image and generated image.
width, height = load_img(base_image_path).size
img_nrows = int(sys.argv[4])
img_ncols = int(width * img_nrows / height)



def preprocess_image(image_path):
	"""
	Generate feature vector of image using VGG16 model.
	"""
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg16.preprocess_input(img)
	return img

def deprocess_image(x):
	if K.image_dim_ordering() == 'th':
		x = x.reshape((3, img_nrows, img_ncols))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((img_nrows, img_ncols, 3))
	# Remove zero-center by mean pixel
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	# 'BGR'->'RGB'
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x



base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))


combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

input_tensor = K.concatenate([base_image,
							  style_reference_image,
							  combination_image], axis=0)



model = vgg16.VGG16(input_tensor=input_tensor,
					weights='imagenet', include_top=False)
model.summary()


outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
for key in outputs_dict:
	print (key, outputs_dict[key])


def gram_matrix(x):
	# features = K.batch_flatten(x)
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

def style_loss(style, combination):
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
	return K.sum(K.square(combination - base))


def total_variation_loss(x):
	assert K.ndim(x) == 4
	
	a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
	b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))



loss = K.variable(0.)
layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
									  combination_features)


# List of feature layers for style image.
feature_layers = ['block1_conv1', 'block2_conv1',
				  'block3_conv1', 'block4_conv1',
				  'block5_conv1']



for layer_name in feature_layers:
	layer_features = outputs_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(combination_image)



grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
	outputs += grads
else:
	outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
	if K.image_dim_ordering() == 'th':
		x = x.reshape((1, 3, img_nrows, img_ncols))
	else:
		x = x.reshape((1, img_nrows, img_ncols, 3))
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')
	return loss_value, grad_values


class Evaluator(object):

	def __init__(self):
		self.loss_value = None
		self.grads_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

evaluator = Evaluator()


# Generates random white noise as inital combined image.
x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128

fig = plt.figure(0)
fig.canvas.set_window_title('Neural Art')

for i in range(iterations):

	print('Start of iteration', i)
	start_time = time.time()
	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
									 fprime=evaluator.grads, maxfun=20)
	print('Current loss value:', min_val)
	
	img = deprocess_image(x.copy())
	fname = directory + '/' + result_prefix + '_iteration_%d.png' % i
	end_time = time.time()
	
	if i == 0 and not os.path.exists(directory):
		os.makedirs(directory)    
		
	imsave(fname, img)
	print('Image saved as', fname)
	print('Iteration %d completed in %ds' % (i, end_time - start_time))
	

	plt.ion()
	plt.imshow(img)
	plt.pause(0.001)








