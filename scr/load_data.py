# -*- coding: UTF-8 -*-

import numpy as np 
import pandas as pd 
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import os,sys,pickle,argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Loading data from images')
parser.add_argument('-p', '--path', type=str, default="/home/garvey18/data/", metavar='dataset path',
                    help='path to dataset')
parser.add_argument('-r', '--resize', type=int, default=int, metavar='resize shape tuple',
                    help='resize the image into this shape while loading')
parser.add_argument('-o', '--output', type=bool, default=True, metavar='pickle',
                    help='output image_array, label as pickle file or not')


def generate_image_path(PATH):
    image_paths=[]
    file_names=[]
    for root,dirs,files in os.walk(PATH,topdown=False):
        if root==PATH+"\guangdong_round1_train1_20180903":
            image_paths+=[os.path.join(root,file) for file in files if ".jpg" in file]
            file_names+=files
        if "guangdong_round1_train2_20180916" in root:
            image_paths+=[os.path.join(root,file) for file in files if file not in file_names and ".jpg" in file]
    return image_paths

def generate_label(image_paths):
    label_replace={"正常":"norm",\
                                    "不导":"defect1",\
                                    "擦花":"defect2",
                                    "横条":"defect3",
                                    "桔皮":"defect4",
                                    "漏底":"defect5",
                                    "碰伤":"defect6",
                                    "起坑":"defect7",
                                    "凸粉":"defect8",
                                    "涂层":"defect9",
                                    "脏点":"defect10",
                                    "伤口":"defect11","划伤":"defect11","变形":"defect11","喷流":"defect11","喷涂":"defect11","打白":"defect11","打磨":"defect11",\
                                    "拖烂":"defect11","杂色":"defect11","气泡":"defect11","油印":"defect11","油渣":"defect11","漆泡":"defect11",\
                                    "火山":"defect11","碰凹":"defect11","粘接":"defect11","纹粗":"defect11","角位":"defect11","返底":"defect11",\
                                    "铝屑":"defect11","驳口":"defect11"}
    labels=[label_replace.get(x.split("/")[-1][:2]) for x in image_paths]
    return labels

def images(image_paths,shape) :
    for v in tqdm(np.asarray(image_paths).reshape(-1)):
        yield resize(image=imread(v),output_shape=shape,mode='constant')



if __name__=='__main__':
    args=vars(parser.parse_args())
    print("------------- Generating Image Path  -------------")
    image_paths = generate_image_path(args["path"])
    print("------------- Generating Label  -------------")
    labels = generate_label(image_paths)
    ohe_label = pd.get_dummies(labels)
    print("------------- Loading Data  -------------")
    images = np.asarray(list(images(image_paths,shape=(args["resize"], args["resize"])   )))
    data= list(zip(images, labels))
    if args["output"]:
        print("------------- Dumping Pickle File  -------------")
        pickle.dump(data,open(args["path"]+"/image_label.pkl","wb"))


