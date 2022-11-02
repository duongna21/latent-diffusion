import os
from notebook_helpers import run
from taming.models import vqgan # checking correct import from taming
import torch
from PIL import Image
import numpy as np

from notebook_helpers import get_model
model = get_model('superresolution')
# print('model: ', model)

custom_steps = 100
# logs = run(model["model"], 'data/example_conditioning/superresolution/celeb64.png', 'superresolution', custom_steps)
logs = run(model["model"], 'data/example_conditioning/superresolution/celeb128.jpg', 'superresolution', custom_steps)
# logs = run(model["model"], 'data/example_conditioning/superresolution/sample_0.jpg', 'superresolution', custom_steps)

sample = logs["sample"]
sample = sample.detach().cpu()
sample = torch.clamp(sample, -1., 1.)
sample = (sample + 1.) / 2. * 255
sample = sample.numpy().astype(np.uint8)
sample = np.transpose(sample, (0, 2, 3, 1))
print(sample.shape)
a = Image.fromarray(sample[0])
a.save('super.jpg')
