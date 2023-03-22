from PIL import Image,ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt

def get_anno(x1,y1,x2,y2,a,c,file):
    l1 = a[:,0] < x2-5
    l2 = a[:,1] < y2-5
    l3 = a[:,2] > x1+5
    l4 = a[:,3] > y1+5
    l12 = np.logical_and(l1,l2)
    l23 = np.logical_and(l3,l4)
    l = np.logical_and(l23,l12)
    a = a[l]
    c = c[l]
    a[:,::2] -= x1
    a[:,1::2] -= y1
    a[:,:2] = (a[:,:2] + a[:,2:])/2
    a[:,2:] = (a[:,2:] - a[:,:2])*2
    a[:,::2] /= (x2-x1)
    a[:, 1::2] /= (y2 - y1)

    with open(file,'w+')as f:
        for c_,a_ in zip(c,a):
            c_ = classnames_v1_5.index(c_)
            f.write(str(c_)+' ')
            for a__ in a_:
                f.write(str(a__)+' ')
            f.write('\n')
def read_txt(txt):
    with open(txt,'r')as f:
        a = f.readlines()[2:]
        c = [i.split(' ')[8] for i in a]
        a = [i.split(' ')[:6] for i in a]
        a = np.array(a,dtype=float)
        a = np.concatenate([a[:,:2],a[:,-2:]],axis=1)
    return a,np.array(c,dtype=object)

# classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
#                    'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank',
#                    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
#                    'container-crane']

classnames_v1_5 = ['small-vehicle','large-vehicle']


ann_file = r'F:\DataSets\DOTA\Vehicle_Split\val\labelTxt'
yolo_file = r'F:\DataSets\DOTA\Vehicle_Split\YOLO\labels\val'

png_file = r'F:\DataSets\DOTA\Vehicle_Split\val\images'
jpg_file = r'F:\DataSets\DOTA\Vehicle_Split\YOLO\images\val'
for file,img in zip(os.listdir(ann_file),os.listdir(png_file)):
    jpg = jpg_file + '\\' + img.replace('png','jpg')
    png = png_file + '\\' + img
    yolo = yolo_file + '\\' + file
    ann = ann_file + '\\' + file
    try:
        a, c = read_txt(ann)
        image = Image.open(png)
        q = 0
        if image.size[0] > 1920 or image.size[1] > 1920:
            for i in range(image.size[0] // 1600):
                for j in range(image.size[1] // 1600):
                    img_ = image.crop((i * 1600, j * 1600, i * 1600 + 1920, j * 1600 + 1920))
                    jpg_ = jpg.replace('.jpg', '%d.jpg' % q)
                    yolo_ = yolo.replace('.txt', '%d.txt' % q)
                    get_anno(i * 1600, j * 1600, i * 1600 + 1920, j * 1600 + 1920, a, c, yolo_)
                    img_.save(jpg_)
                    q += 1
                img_ = image.crop((i * 1600, image.size[1] - 1920, i * 1600 + 1920, image.size[1]))
                jpg_ = jpg.replace('.jpg', '%d.jpg' % q)
                yolo_ = yolo.replace('.txt', '%d.txt' % q)
                get_anno(i * 1600, image.size[1] - 1920, i * 1600 + 1920, image.size[1], a, c, yolo_)
                img_.save(jpg_)
                q += 1
            for j in range(image.size[1] // 1600):
                img_ = image.crop((image.size[0] - 1920, j * 1600, image.size[0], j * 1600 + 1920))
                jpg_ = jpg.replace('.jpg', '%d.jpg' % q)
                yolo_ = yolo.replace('.txt', '%d.txt' % q)
                get_anno(image.size[0] - 1920, j * 1600, image.size[0], j * 1600 + 1920, a, c, yolo_)
                img_.save(jpg_)
                q += 1
            img_ = image.crop((image.size[0] - 1920, image.size[1] - 1920, image.size[0], image.size[1]))
            jpg_ = jpg.replace('.jpg', '%d.jpg' % q)
            yolo_ = yolo.replace('.txt', '%d.txt' % q)
            get_anno(image.size[0] - 1920, image.size[1] - 1920, image.size[0], image.size[1], a, c, yolo_)
            img_.save(jpg_)
            q += 1
        else:
            # jpg_ = jpg.replace('.jpg', '%d.jpg' % q)
            # yolo_ = yolo.replace('.txt', '%d.txt' % q)
            get_anno(0, 0, image.size[0], image.size[1], a, c, yolo)
            image.save(jpg)
        # print(img + ' finished!')
    except:
        print(img + 'failed')

