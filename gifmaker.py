import os
from PIL import Image

save_dir = 'detected/rotated-tree/'

images = []
files = os.listdir(save_dir)
for file_name in files:
    _, ext = os.path.splitext(file_name)
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        im = Image.open(os.path.join(save_dir, file_name))
        images.append(im)

images[0].save('tree.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
