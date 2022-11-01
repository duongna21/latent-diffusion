import os
from notebook_helpers import run
from taming.models import vqgan # checking correct import from taming
from notebook_helpers import get_model
model = get_model(mode.value)

custom_steps = 100
cond_choice_path = os.path.join(dir, )
logs = run(model["model"], 'data/example_conditioning/superresolution/sample_0.jpg', mode.value, custom_steps)