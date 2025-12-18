
import torch
import numpy as np
from utils import model, helper_functions
from torchvision.transforms.functional import to_pil_image
import os
import argparse

default_img_path = "imgs/test_img1.jpg"
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=default_img_path)
args = parser.parse_args()
img_path = args.image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_URL = "https://github.com/LostInDimensions/CNN_colorizer/releases/download/v1.0/colorizer_release.pth"
MODEL_PATH = "colorizer_release.pth"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found. Downloading model from {MODEL_URL}...")
    torch.hub.download_url_to_file(MODEL_URL, MODEL_PATH)
    print("Download finished")
else:
    print("model already exists, skipping download.")

model = model.UNetColorizationNet().eval().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))

PTS_IN_HULL_PATH = "utils/pts_in_hull.npy"
pts_in_hull = np.load(PTS_IN_HULL_PATH)
pts = torch.from_numpy(pts_in_hull).float().to(DEVICE)

print(f"Starting colorization of '{img_path}'...")

resized_img = helper_functions.resize_img(img_path)
rgb_pred = helper_functions.make_rgb_pred(resized_img, pts, model)
PIL_img = to_pil_image(rgb_pred)
img_out_path = "out_imgs/colorized_" + img_path.split("/")[-1]
PIL_img.save(img_out_path)
print(f"colorized image was saved under: {img_out_path}")
