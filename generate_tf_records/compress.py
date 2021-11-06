from PIL import Image
import os
import glob


images_to_compress = glob.glob("../images/*.jpg")
images_to_compress.sort()

if not os.path.exists('compressed'):
    os.makedirs('compressed')

counter = 1
for image_path in images_to_compress:
    print("Compressing image " + str(counter))
    img = Image.open(image_path)
    if img.size == (4032,3024):
        img = img.resize((640,480),Image.ANTIALIAS)
    else:
        img = img.resize((480,640),Image.ANTIALIAS)
        img = img.rotate(90,expand=True)
    img.save('compressed/'+str(counter).zfill(4)+'.jpg',quality=95)
    counter = counter + 1
