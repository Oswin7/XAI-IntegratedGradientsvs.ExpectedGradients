import numpy as np
from integrated_gradients.integrated_gradients import compute_integrated_gradients
from datasets import load_dataset
from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
import random

def expected_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    num_baselines,
    steps=50
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
    see integrated_gradients
    except num_baselines: number of baselines

    '''
    train_ds = ImageFolder(
        root="imagenette2-320\\imagenette2-320\\train"
    )   
    
    img, label = train_ds[random.randrange(len(train_ds))]

    integrated_gradients_list = None
    predictions_list = None

    for i in range(0, num_baselines):
        img, _ = train_ds[random.randrange(len(train_ds))]
        baseline = img.convert("RGB").resize((224,224))
        baseline_tensor = transforms.ToTensor()(baseline)
        ig, pred = compute_integrated_gradients(inp, target_label_index, predictions_and_gradients, baseline_tensor, steps)
        if integrated_gradients_list is None:
            integrated_gradients_list = ig.unsqueeze(0)
            predictions_list = pred.unsqueeze(0)
        else:
            integrated_gradients_list = torch.cat((integrated_gradients_list, ig.unsqueeze(0)), dim=0)
            predictions_list = torch.cat((predictions_list, pred.unsqueeze(0)), dim=0)
        
    expected_gradients = torch.mean(integrated_gradients_list, dim=0)
    predictions_eg = torch.mean(predictions_list, dim=0)

    return expected_gradients, predictions_eg