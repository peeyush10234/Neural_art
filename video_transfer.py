
# coding: utf-8

# In[2]:

from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

from keras.applications import vgg16 , InceptionV3
from keras import backend as K
import cv2

img_nrows = 0
img_ncols = 0
f_outputs = []


# In[3]:


# vc = cv2.VideoCapture('/home/salvation/fox.mp4')
# c=1

# if vc.isOpened():
#     rval , frame = vc.read()
# else:
#     rval = False

# while rval:
#     rval, frame = vc.read()
#     cv2.imwrite(str(c) + '.jpg',frame)
#     c = c + 1
#     cv2.waitKey(1)
# vc.release()


# In[4]:

def preprocess_image(image_path):
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


# In[5]:

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


# In[6]:

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


# In[7]:

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self,x):
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


# In[8]:

def neural_art(base_img,style_img,epochs,j):
    base_image_path = base_img
    style_reference_image_path = style_img
    result_prefix = 'im'
    iterations = epochs

    total_variation_weight = 1.0
    style_weight = 0.8
    content_weight = 0.125
    
    width, height = load_img(base_image_path).size
    global img_nrows
    global img_ncols
    img_nrows = 300
    img_ncols = int(width * img_nrows / height)
    
    base_image = K.variable(preprocess_image(base_image_path))
    style_reference_image = K.variable(preprocess_image(style_reference_image_path))
    
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

    input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)
    
    model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
    
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    loss = K.variable(0.)
    layer_features = outputs_dict['block4_conv2']
#   print layer_features.shape
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                      combination_features)
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
    
    global f_outputs
    f_outputs = K.function([combination_image], outputs)
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
    
        img = deprocess_image(x.copy())
        fname = '/output/im' + '_at_iteration_%d.png' % j
        if i == 14:
            imsave(fname, img)
        
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
    
    
    


# In[9]:

for i in range(1,420):
    base_path1 = '/input/%d.jpg'%i
    neural_art(base_path1,'/input/la_muse.jpg', 20,i)


# In[3]:




# In[ ]:



