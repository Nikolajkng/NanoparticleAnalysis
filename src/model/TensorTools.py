from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def crop(tensor: Tensor, target_size: tuple[int, int]) -> Tensor:
    _, _, h, w = tensor.shape
    th, tw = target_size

    start_h = (h-th) // 2
    start_w = (w-tw) // 2
    return tensor[:, :, start_h:start_h + th, start_w:start_w + tw]

def normalizeTensorToPixels(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255
    return tensor

def showTensor(tensor: Tensor) -> None:
    probabilities = F.softmax(tensor, dim=1)  
    probabilities = probabilities.squeeze(0)
    
    pixels = normalizeTensorToPixels(probabilities[1, :, :])
    
    img = TF.to_pil_image(pixels.byte())
    img.show()