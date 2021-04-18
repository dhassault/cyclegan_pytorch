"""
From the paper, 4. Implemention, Training details:
"[...] to  reduce  model  oscillation  [15], 
we  follow Shrivastava  et  al.’s  strategy  [46] 
and  update  the  discriminators using a history
of generated images rather than the ones produced
by the latest generators.  We keep an image buffer
that stores the 50 previously created images."
"""
import random

import torch
from torch.autograd import Variable


class ImageBuffer:
    def __init__(self, size=50):
        self.size = size
        self.data = []

    def push_and_pop(self, data):
        images_arr = []
        for image in data.data:
            image = torch.unsqueeze(image, 0)
            if len(self.data) < self.size:
                self.data.append(image)
                images_arr.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.size - 1)
                    images_arr.append(self.data[i].clone())
                    self.data[i] = image
                else:
                    images_arr.append(image)
        return Variable(torch.cat(images_arr))
