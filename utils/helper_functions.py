
import torch
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import torch.nn.functional as F

def annealed_mean_from_probs(probs, pts):
    probs = probs ** 2.5
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
    ab = torch.einsum('nchw,cd->ndhw', probs, pts)
    return ab


def resize_img(img_path):
    rgb_image = Image.open(img_path)
    w, h = rgb_image.size

    for i in range(64):
        if h % 32 != 0:
            h = h - 1
        elif w % 32 !=0:
            w = w - 1
        else:
            break
    resized_img = rgb_image.resize(size=(w, h))

    return resized_img

def make_rgb_pred(rgb_img, pts, model):
    lab_img = rgb2lab(rgb_img)
    L_in = torch.from_numpy(lab_img).permute(2, 0, 1)[0]
    L_in_t = (L_in - 50) / 100

    with torch.inference_mode():
        ab_probs = model(L_in_t.unsqueeze(0).unsqueeze(0).float())
        ab_probs = F.softmax(ab_probs, dim=1)
        
    ab_preds = annealed_mean_from_probs(ab_probs, pts)
    lab_pred = torch.cat((L_in.unsqueeze(0), ab_preds.squeeze()))
    rgb_pred = lab2rgb(lab_pred.permute(1, 2, 0))

    return rgb_pred