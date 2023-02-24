import patchify 
import torch
import numpy as np
data=np.array([i for i in range(25)]).reshape(1,5,5)
print(data.shape)
patched=patchify.patchify(data,(1,2,2),step=(2,2))
patched=patched.reshape(-1,2,2)
print(patched)
print(patched.shape)