import os
from PIL import Image

source_images_path = '/Volumes/Noah/GOPRO_Large_all/train'
save_path = '/Volumes/Noah/gopro_combined_cropped/'
count = 0
for root, dirs, files in os.walk(source_images_path):
    path = root.split(os.sep)
    #print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        if file[len(file)-1] == 'g' and file[0]!= '.' and count < 100:
            count = count + 1
        print(count)
        im = Image.open('/'.join(path)+'/'+file) #Can be many different formats.
        im.crop((0, 0, 720, 720)).save(save_path+'unblurred'+str(count)+'.png')