from integrated_gradients.integrated_gradients import compute_integrated_gradients
from model.resnet50_utils import make_predictions_and_gradients
from PIL import Image
import torch
import numpy as np
from expected_gradients.expected_gradients import compute_expected_gradients
from datasets import load_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import models, transforms
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

def get_image() -> Image:
    train_ds = ImageFolder(
        root= ROOT / "imagenette2-320" / "imagenette2-320" / "train"
    )

    img, label = train_ds[random.randrange(len(train_ds))]
    return img

def gaussian_perturbation(img: Image.Image, sigma: float = 0.01) -> Image.Image:
    """
    Fügt Gaussian Noise hinzu.
    sigma bezieht sich auf Bilder im Bereich [0,1].
    """
    x = TF.to_tensor(img)

    noise = torch.randn_like(x) * sigma
    x = (x + noise).clamp(0, 1)

    return TF.to_pil_image(x)

def brightness_perturbation(img: Image.Image, brightness: float = 1.1) -> Image.Image:
    """
    brightness >1 heller, <1 dunkler.
    z.B. 1.05 oder 0.95
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness)

def translation_perturbation(
    img: Image.Image,
    trans_x: int = 3,
    trans_y: int = 3
) -> Image.Image:
    """
    Verschiebt das Bild um trans_x und trans_y Pixel.
    """
    return TF.affine(
        img,
        angle=0,
        translate=(trans_x, trans_y),
        scale=1.0,
        shear=0,
        fill=0
    )



def similarity_pearson_corr(
    attribution: np.ndarray,
    perturbated_attribution: np.ndarray
) -> float:

    a = attribution.abs().mean(dim=0).flatten().cpu().numpy()
    b = perturbated_attribution.abs().mean(dim=0).flatten().cpu().numpy()

    corr, _ = pearsonr(a, b)
    return corr

def similarity_ssim(
    attribution: torch.Tensor,
    perturbated_attribution: torch.Tensor
) -> float:

    attribution = attribution.detach().cpu().numpy()
    perturbated_attribution = perturbated_attribution.detach().cpu().numpy()

    return ssim(
        attribution,
        perturbated_attribution,
        data_range=attribution.max() - attribution.min(),
        channel_axis=0
    )

def similarity_cosine_similarity(
    attribution: np.ndarray,
    perturbated_attribution: np.ndarray
) -> float:

    a = attribution.flatten().reshape(1, -1)
    b = perturbated_attribution.flatten().reshape(1, -1)

    return cosine_similarity(a, b)[0, 0]



def compute_stability(img: Image.Image, target_label_index: int = 0, baseline = None, num_baselines: int = 1, num_steps: int = 25, plot_name = "plot.png"):
    img = img.convert("RGB").resize((224,224))

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

    gaussian_noise = 0.1
    brightness_pert = 2
    translation_x = 10
    translation_y = 10
    perturbations = [transforms.ToTensor()(gaussian_perturbation(img, gaussian_noise)), transforms.ToTensor()(brightness_perturbation(img, brightness_pert)), transforms.ToTensor()(translation_perturbation(img, translation_x, translation_y))]
    
    eg_perturbation_probs = []
    gradients_eg_perturbated = []
    predictions_eg_perturbated = []

    input_tensor = transforms.ToTensor()(img)
    
    gradients_eg, predictions_eg = compute_expected_gradients(input_tensor, target_label_index, make_predictions_and_gradients, num_baselines, num_steps)
    eg_max_class = torch.argmax(predictions_eg[num_steps - 1])
    eg_max_class_number = eg_max_class.item()
    eg_label = labels[eg_max_class_number]
    eg_prob = predictions_eg[num_steps - 1][eg_max_class_number].item()
    print("expected gradient for original image finished")
    for i, perturbation in enumerate(perturbations):
        grad, pred = compute_expected_gradients(perturbation, target_label_index, make_predictions_and_gradients, num_baselines, num_steps)
        gradients_eg_perturbated.append(grad)
        predictions_eg_perturbated.append(pred)
        eg_prob = pred[num_steps - 1][eg_max_class_number].item()
        eg_perturbation_probs.append(eg_prob)
        print("perturbation eg number ", i, " done")


    similarities_eg = []

    for perturbed in gradients_eg_perturbated:
        similarities_eg.append({
            "pearson": similarity_pearson_corr(gradients_eg, perturbed),
            "ssim": similarity_ssim(gradients_eg, perturbed),
            "cosine": similarity_cosine_similarity(gradients_eg, perturbed)
        })


    gradients_ig_perturbated = []
    predictions_ig_perturbated = []
    ig_perturbation_probs = []
    gradients_ig, predictions_ig = compute_integrated_gradients(input_tensor, target_label_index, make_predictions_and_gradients, None, num_steps)
    ig_max_class = torch.argmax(predictions_ig[num_steps - 1])
    ig_max_class_number = ig_max_class.item()
    ig_label = labels[ig_max_class_number]
    ig_prob = predictions_ig[num_steps - 1][ig_max_class_number].item()
    print("integrated gradient for original image finished")
    for i, perturbation in enumerate(perturbations):
        grad, pred = compute_integrated_gradients(perturbation, target_label_index, make_predictions_and_gradients, None, num_steps)
        gradients_ig_perturbated.append(grad)
        predictions_ig_perturbated.append(pred)
        ig_prob = pred[num_steps - 1][ig_max_class_number].item()
        ig_perturbation_probs.append(ig_prob)
        print("perturbation ig number ", i, " done")

    similarities_ig = []

    for perturbed in gradients_ig_perturbated:
        similarities_ig.append({
            "pearson": similarity_pearson_corr(gradients_ig, perturbed),
            "ssim": similarity_ssim(gradients_ig, perturbed),
            "cosine": similarity_cosine_similarity(gradients_ig, perturbed)
        })

    print("similarities expected gradients: ", similarities_eg)
    print("similarities integrated gradients: ", similarities_ig)


    plot_array = np.empty((3,len(perturbations) + 1), dtype=object)
    rows, cols = plot_array.shape
    plot_array[0, 0] = input_tensor
    ig_gradient = (gradients_ig.abs() - gradients_ig.abs().min()) / (gradients_ig.abs().max() - gradients_ig.abs().min())
    plot_array[1, 0] = ig_gradient
    eg_gradient = (gradients_eg.abs() - gradients_eg.abs().min()) / (gradients_eg.abs().max() - gradients_eg.abs().min())
    plot_array[2, 0] = eg_gradient
    for i in range(1, len(perturbations) + 1):
        plot_array[0, i] = perturbations[i-1] # perturbated images
        ig_gradient = gradients_ig_perturbated[i-1]
        plot_array[1, i] = (ig_gradient.abs() - ig_gradient.abs().min()) / (ig_gradient.abs().max() - ig_gradient.abs().min()) # integrated gradients
        eg_gradient = gradients_eg_perturbated[i-1]
        plot_array[2, i] = (eg_gradient.abs() - eg_gradient.abs().min()) / (eg_gradient.abs().max() - eg_gradient.abs().min()) # expected gradients


    fig, axs = plt.subplots(3, 4)

    row_labels = ["image", "integrated gradients", "expected Gradients"]
    for i in range(rows):
        axs[i, 0].set_ylabel(row_labels[i], fontsize=12)
    column_labels = ["original", "gaussian noise", "brightness", "transition"]
    for i in range(cols):
        axs[0, i].set_xlabel(column_labels[i], fontsize=12)


    axs[0, 0].set_title("Original", fontsize=4) # original
    axs[1, 0].set_title(f"Integrated Gradients\nclass: {ig_label}\nprob: {ig_prob:.3f}", fontsize=4)
    axs[2, 0].set_title(f"Expected Gradients\nclass: {eg_label}\nprob: {eg_prob:.3f}", fontsize=4)
    axs[0, 1].set_title(f"Noise: {gaussian_noise}", fontsize=4) # gaussian noise
    axs[1, 1].set_title(f"pearson-similarity: {float(similarities_ig[0]["pearson"]):.4f}\nprob: {ig_perturbation_probs[0]:.3f}", fontsize=4)
    axs[2, 1].set_title(f"pearson-similarity: {float(similarities_eg[0]["pearson"]):.4f}\nprob: {eg_perturbation_probs[0]:.3f}", fontsize=4)
    axs[0, 2].set_title(f"pearson-Brightness factor {brightness_pert}", fontsize=4) # brightniss
    axs[1, 2].set_title(f"pearson-similarity: {float(similarities_ig[1]["pearson"]):.4f}\nprob: {ig_perturbation_probs[1]:.3f}", fontsize=4)
    axs[2, 2].set_title(f"pearson-similarity: {float(similarities_eg[1]["pearson"]):.4f}\nprob: {eg_perturbation_probs[1]:.3f}", fontsize=4)
    axs[0, 3].set_title(f"x: {translation_x}, y: {translation_y}", fontsize=4) # translation
    axs[1, 3].set_title(f"pearson-similarity: {float(similarities_ig[2]["pearson"]):.4f}\nprob: {ig_perturbation_probs[2]:.3f}", fontsize=4)
    axs[2, 3].set_title(f"pearson-similarity: {float(similarities_eg[2]["pearson"]):.4f}\nprob: {eg_perturbation_probs[2]:.3f}", fontsize=4)

    for i in range(rows): # 3
        for j in range(cols): # 4
            axs[i, j].set_axis_off()
            if i == 0:
                axs[i, j].imshow(plot_array[i, j].permute(1,2,0))
            else:
                axs[i, j].imshow(plot_array[i, j].permute(1,2,0), cmap="hot")

    path = ROOT / "evaluation" / "stability" / "images" / plot_name
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_target_label(img: Image.Image) -> int:
    # 1. Load pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
    	transforms.Resize(256),
		transforms.CenterCrop(224),
        transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		)
    ])

    input_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    pred_class = logits.argmax(dim=1).item()

    return pred_class

def stabilty_test(seed = -1, num_baselines = 2, num_steps = 10, plot_name = "plot.png"):
    print("starting")
    if seed != -1:
        random.seed(seed)
    img = get_image()
    label = get_target_label(img)
    print("label: ", label)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urllib.request.urlopen(url).read().decode("utf-8").split("\n")
    print(labels[label])
    path = ROOT / "evaluation" / "stability" / "images" / "image.png"
    img.save(path)
    print("start perturbation")
    compute_stability(img, label, baseline = None, num_baselines = num_baselines, num_steps = num_steps, plot_name = plot_name)
    print("done")

if __name__ == "__main__":
    for i in range (10):
        stabilty_test(seed = i, num_baselines = 5, num_steps = 10, plot_name = f"run{i}_plot_{i}_baselines5_steps10.png")
#    for i in range (10):
#        stabilty_test(seed = -1, num_baselines = 100, num_steps = 25, plot_name = f"run1{i}_plot_seed42_baselines100_steps25.png")
    stabilty_test(seed = 42, num_baselines = 5, num_steps = 10, plot_name = "plot3.png")

    # baseline ig = black/white/grey
    # num_baselines 10, 100
    # num_steps = 25