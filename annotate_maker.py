import pandas as pd
import numpy as np


train = pd.read_csv('train.csv')
flip = pd.read_csv('FLIP.csv')
flip_rotate = pd.read_csv('FLIP_ROTATE.csv')
flip_scale = pd.read_csv('FLIP_SCALE.csv')
flip_shear = pd.read_csv('FLIP_SHEAR.csv')
flip_translate = pd.read_csv('FLIP_TRANSLATE.csv')
rotation = pd.read_csv('ROTATION.csv')
scaled = pd.read_csv('SCALED.csv')
shear = pd.read_csv('SHEAR.csv')
translated = pd.read_csv('TRANSLATED.csv')
det_scale = pd.read_csv('DET_SCALE.csv')

flip.columns = train.columns
train = train.append(flip, ignore_index = True)


flip_rotate.columns = train.columns
train = train.append(flip_rotate, ignore_index = True)

flip_scale.columns = train.columns
train = train.append(flip_scale, ignore_index = True)

flip_shear.columns = train.columns
train = train.append(flip_shear, ignore_index = True)


flip_translate.columns = train.columns
train = train.append(flip_translate, ignore_index = True)

rotation.columns = train.columns
train = train.append(rotation, ignore_index = True)

scaled.columns = train.columns
train = train.append(scaled, ignore_index = True)

shear.columns = train.columns
train = train.append(shear, ignore_index = True)

translated.columns = train.columns
train = train.append(translated, ignore_index = True)

det_scale.columns = train.columns
train = train.append(det_scale, ignore_index = True)


a = train.iloc[:, :].values

for i in range(0, len(a)):
    a[i][1] = int(a[i][1])
    a[i][2] = int(a[i][2])
    a[i][3] = int(a[i][3])
    a[i][4] = int(a[i][4])

column_list = list(train.columns)
train = pd.DataFrame(a)
train.columns = column_list

data = pd.DataFrame()
data['format'] = train['filename']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'train_images/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + str(train['label'][i])


data.to_csv('annotate.txt', header = None, index = None)
