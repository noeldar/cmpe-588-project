import torch, torchvision
from torch import nn
from torchvision import transforms, models, datasets
import shap
import json
import numpy as np
import utils
import pickle

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float().to(device='cuda')

def normalize_v2(image):
    """if image.max() > 1:
        image /= 255
    image = (image - mean) / std"""
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float().to(device='cuda')


model = models.vgg16(pretrained=True).eval()

X, y = shap.datasets.imagenet50()

X /= 255

to_explain = X[[39, 41]]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

if torch.cuda.is_available():
    #X = X.to('cuda')
    #to_explain = to_explain.to('cuda')
    model.to('cuda')
e = shap.GradientExplainer((model, model.features[7]), normalize(X))

batch_shape = [20, 224, 224, 3]
image_preprocessing_fn = utils.normalization_fn_map["vgg_16"]
count=0
for images, names, labels in utils.load_image('./dataset/images/', 224, 20):
    count += 20
    if count % 100 == 0:
        print("Generating:", count)

    """images_tmp = image_preprocessing_fn(np.copy(images))
    images_adv = images
    images_adv = image_preprocessing_fn(np.copy(images_adv))
    images_tmp2 = image_preprocessing_fn(np.copy(images))
    print(count)
    shap_values, indexes = e.shap_values(normalize_v2(images_tmp2), ranked_outputs=2, nsamples=200)

    with open('/media/tanlab/2TB/nomask_'+str(count)+'.pickle', 'wb') as f:
        pickle.dump(shap_values, f)"""

    for l in range(int(5)):
        # generate the random mask
        mask = np.random.binomial(1, 0.7, size=(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))
        images_tmp2 = images * mask
        images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))

        shap_values, indexes = e.shap_values(normalize_v2(images_tmp2), ranked_outputs=2, nsamples=200)

        with open('/media/tanlab/2TB/withmask_' + str(count) +"_"+str(l)+'.pickle', 'wb') as f:
            pickle.dump(shap_values, f)
