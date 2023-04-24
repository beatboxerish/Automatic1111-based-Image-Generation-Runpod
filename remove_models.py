import os

print(os.listdir("/workspace/models/ldm/stable-diffusion-v1"))
os.remove("/workspace/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt")
os.remove("/workspace/models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt")
