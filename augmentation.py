from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('train.csv')

a = train.iloc[:, :].values

h = 400   # number of unique images in train
Matrix = [[] for y in range(h)] 

# segregrating entries per image
k = 0
img_name = []
img_name.append(a[0][0])
Matrix[k].append(a[0][1:6])
for i in range(1, len(a)):
    if a[i][0] == a[i-1][0]:
        Matrix[k].append(a[i][1:6])
    else:
        k = k+1
        Matrix[k].append(a[i][1:6])
        img_name.append(a[i][0])

#Matrix[0], Matrix[1] etc are my bounding boxes


########### flipping
k = 401

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)
    
    transforms = Sequence([RandomHorizontalFlip(1)])
    img, bboxes = transforms(img, bboxes)
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP.csv', index = False)
    

############ scaling
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    scale = RandomScale(0.2, diff = True)
    img,bboxes = scale(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('SCALED.csv', index = False)
    
    
    
#####Translation
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    translate = RandomTranslate(0.4, diff = True)
    img, bboxes = translate(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('TRANSLATED.csv', index = False)
    
    
######rotation
   
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    rotate = RandomRotate(10)  ## rotating by 10 degrees
    img, bboxes = rotate(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('ROTATION.csv', index = False)
    
    
    
########################Shearing
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    shear = RandomShear(0.7)  
    img, bboxes = shear(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('SHEAR.csv', index = False)
    
    
    
#########deterministic scaling
    
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    scale = RandomScale((0.4))  
    img, bboxes = scale(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('DET_SCALE.csv', index = False)
    
    
    
#########flip and scale
k = k+1

for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_SCALE.csv', index = False)
    
    

##### flip and translation
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomTranslate(0.3, diff = True)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_TRANSLATE.csv', index = False)
    
###########flip and rotation
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomRotate(20)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_ROTATE.csv', index = False)

###################### flip and Shear
k=k+1
for i in range(0, len(img_name)):
    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
    bboxes = Matrix[i]
    bboxes = np.asarray(bboxes)

    transforms = Sequence([RandomHorizontalFlip(1), RandomShear(0.6)])
    img, bboxes = transforms(img, bboxes)
    
    if i == 0:
        bounding_boxes = pd.DataFrame(bboxes)
        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
    
    if i>0:
        df = pd.DataFrame(bboxes)
        k = k+1
        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
    
    from PIL import Image
    img = Image.fromarray(img)
    img.save('img{}.png'.format(k))
    print("img{}.png =-------= {}".format(k, i))
    bounding_boxes.to_csv('FLIP_SHEAR.csv', index = False)
    

######################flip and scale and rotate
#k=k+1
#for i in range(0, len(img_name)):
#    img = cv2.imread("{}".format(img_name[i]))[:,:,::-1]
#    bboxes = Matrix[i]
#    bboxes = np.asarray(bboxes)
#
#    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.3, diff = True), RandomRotate(30)])
#    img, bboxes = transforms(img, bboxes)
#    
#    if i == 0:
#        bounding_boxes = pd.DataFrame(bboxes)
#        bounding_boxes.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
#    
#    if i>0:
#        df = pd.DataFrame(bboxes)
#        k = k+1
#        df.insert(loc = 0, column = 'image_name', value = 'img{}.png'.format(k))
#        bounding_boxes = bounding_boxes.append(df, ignore_index = True)
#    
#    from PIL import Image
#    img = Image.fromarray(img)
#    img.save('img{}.png'.format(k))
#    print("img{}.png =-------= {}".format(k, i))
#    bounding_boxes.to_csv('FLIP_SCALE_ROTATE.csv', index = False)
