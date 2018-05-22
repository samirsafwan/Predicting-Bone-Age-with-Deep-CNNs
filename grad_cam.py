from PIL import Image

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F 
import numpy as np 
import skimage.transform 
import model.net as net
import utils

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

image = Image.open("gc.png").convert('RGBA')
'''
imshow(image)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image)


plt.show()
'''
preprocess = transforms.Compose([
    transforms.Grayscale(3),
    transforms.ToTensor()
    ])

display_transform = transforms.Compose([
    transforms.Resize((299,299))
    ])

gender = torch.zeros([1,0])
gender = Variable(gender, requires_grad = True)
gender = gender.view(gender.shape[0],1)

print("preprocessing image to torch tensor")
tensor = preprocess(image)
image_var = Variable((tensor.unsqueeze(0)), requires_grad = True)
params = utils.Params('model_5_21/params.json')
model = net.Net(params)
utils.load_checkpoint('model_5_21/best.pth.tar', model)
model.eval()

class SaveFeatures():
    features = None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features=output.data.cpu().numpy()
    def remove(self): 
        self.hook.remove()

print("getting activated features")

# USE ONE OF THESE
#activated_features = SaveFeatures(model.model.Mixed_7c.branch1x1) #or fc3


#activated_features = SaveFeatures(model.model.Mixed_7c.branch3x3dbl_3b)
#activated_features = SaveFeatures(model.model.Conv2d_4a_3x3)

# USE THIS
#activated_features = SaveFeatures(model.model.Mixed_6e.branch7x7_3)



prediction = model((image_var, gender)).data.squeeze()
activated_features.remove()

def getCAM(feature_conv, weight_fc):
    _, nc, h, w = feature_conv.shape
    print("doing dot product")
    cam = np.dot(weight_fc, feature_conv.reshape((nc, h*w)))
    print("reshaping")
    cam = cam.reshape(h,w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

print("getting weights")

# USE THIS
'''
weight = list(model.model.Mixed_6e.branch7x7_3.parameters())
weight = np.squeeze(weight[len(weight)-2].data.cpu().numpy())
'''

# USE ONE OF THESE
'''
weight = list(model.model.Mixed_7c.branch1x1.parameters())
weight = np.squeeze(weight[len(weight)-2].data.cpu().numpy())
'''

'''
weight = list(model.model.Mixed_7c.branch3x3dbl_3b.parameters())
weight = np.squeeze(weight[len(weight)-2].data.cpu().numpy())
'''

# MAYBE DONT USE THIS
'''
weight = list(model.model.Conv2d_4a_3x3.parameters())
weight = np.squeeze(weight[len(weight)-2].data.cpu().numpy())
'''

print("getting overlay")
overlay = getCAM(activated_features.features, weight)

imshow(overlay[0], alpha=0.5, cmap='jet')
imshow(display_transform(image))
imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
plt.show()



