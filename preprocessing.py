import numpy as np
from glob import glob
from PIL import Image
import random
import sys
import matplotlib.pyplot as plt
import os

class PREPROCESSING:
    def __init__(self, title, imgSize):
        self.title = title
        self.imgSrc = 'DCGAN\\hyunjun\\DATA\\images\\' + self.title + '\\*.jpg'
        self.imgSize = imgSize

        self.datalist = glob(self.imgSrc)
        self.length = len(self.datalist)

    def getImage(self):
        img = Image.open(self.datalist[random.randint(0, self.length - 1 )])
        img = np.array(img.resize((self.imgSize, self.imgSize)))

        try:
            result = []
            for y in range(len(img)):
                tmp=[]
                for x in range(len(img[y])):
                    tmp.append((img[y][x][0:3] / 127.5) - 1)
                    # tmp.append(img[y][x][0:3] / 255)
                result.append(tmp)
            img = result
        except Exception as e:
            print('error : ' , e)
            return False
        return img
    def getRandomBatch(self, batchSize):
        result=[]
        for _ in range(batchSize):
            result.append(self.getImage())
        return result

    def saveImgae(self, img1, img2, title):
        dir = "DATA\\Results\\"+self.title
        if not os.path.exists(dir):
            os.makedirs(dir)
        image = np.concatenate((img1, img2), axis=1)
        plt.imshow(image, cmap='gray')
        plt.savefig(dir + '\\' + str(title)+'.jpg')
