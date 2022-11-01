import os
from notebook_helpers import run
from taming.models import vqgan # checking correct import from taming
from notebook_helpers import get_model
model = get_model('superresolution')
print('model: ', model)

custom_steps = 1
logs = run(model["model"], 'data/example_conditioning/superresolution/sample_0.jpg', 'superresolution', custom_steps)