import torch
from torchvision import models, transforms
from PIL import Image
from typing import List
import numpy as np

def make_predictions_and_gradients(images: List[torch.Tensor], target_label_index):
	"""Returns a function that can be used to obtain the predictions and gradients
	from the Inception network for a set of inputs. 

	The function is meant to be provided as an argument to the integrated_gradients
	method.
	"""
	# 1. Load pretrained model
	model = models.resnet50(pretrained=True)
	model.eval()

	# 2. Image preprocessing (VERY important)
	preprocess = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
	])

	# 3. Load image
	# img = Image.open("imgs/dog2.jpg").convert("RGB")
	images_tensor = []
	for image in images:
		images_tensor.append(preprocess(image))  # add batch dimension


	# 4. Inference

	images_tensor = torch.stack(images_tensor).requires_grad_(True)

	output = model(images_tensor)

	# 5. Get predicted class
	preds = torch.nn.functional.softmax(output, dim=1)
	pred_target_labels = preds[:,target_label_index]
	model.zero_grad()

	print(pred_target_labels.shape)
	pred_target_labels.sum().backward()

	print(images_tensor.shape)
	grads = images_tensor.grad

	print(grads.shape)

	return preds, grads


# make_predictions_and_gradients([Image.open("..\\Images\\1bd6987fa9219dec.jpg").convert("RGB")], 0)