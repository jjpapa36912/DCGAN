import preprocessing as pr
import tensorflow as tf
import numpy as np
import model as md
import random
import os


# title = 'cezanne'
# title = 'apple'
title = 'face'
saveOpt = ""
noiseSize = 100
batch_size = 10
LoadModel = False
model = md.Model({
    "noiseSize" : noiseSize
})

'''
try:
    for m in range(len(model.models)):
        dir = 'DATA\\Weights\\' + title + '\\' + saveOpt + '\\'
        if not os.path.exists(dir):
            os.makedirs(dir)
        model.models[m] = tf.keras.models.load_model(
            dir + model.models[m].name + '.h5')
    if LoadModel:
        model.modelLoad()
        print("Load Models")
except:
    model = md.Model({
        "noiseSize": noiseSize
    })
    print("Create New Model")
'''

dataset = pr.PREPROCESSING(title, model.generator.output_shape[1])

def genLable(batch, l , n):
    result=[]
    for _ in range(batch):
        tmp=[0 for _ in range(l)]
        tmp[n] = 1
        result.append(tmp)
    return np.array(result)

def argmax(arr1, arr2):
    
    arr1 = np.argmax(arr1, axis=1)
    arr2 = np.argmax(arr2, axis=1)

    return arr1, arr2


def accuracy(arr1, arr2):
    # print('======', arr1, '=======', arr2)
    # arr1, arr2 = argmax(arr1, arr2)
    for i in range(len(arr1)):
        if arr1[i] < 0.5:
            arr1[i] = 0
        else:
            arr1[i] = 1
    print('======', arr1, '=======', arr2)
    sum = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i]:
            sum += 1
    return sum / len(arr1)


def shuffle(arr1, arr2):
    for i in range(len(arr1)):
        rand = random.randint(0, len(arr1) - 1)
        rand2 = random.randint(0, len(arr1) - 1)
        if rand == rand2:
            continue
        tmp1 = arr1[rand2]
        tmp2 = arr2[rand2]

        arr1[rand2] = arr1[rand]
        arr2[rand2] = arr2[rand]

        arr1[rand] = tmp1
        arr2[rand] = tmp2
    return np.array(arr1), np.array(arr2)


def train(epochs):
    print("Train Start")
    # model.compile()
    model.discriminator.compile(tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), tf.keras.losses.binary_crossentropy)
    model.combine.compile(tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5), tf.keras.losses.binary_crossentropy)
    # fakes = genLable(batch_size, 2, 0)
    # reals = genLable(batch_size, 2, 1)
    # print(fakes.shape, reals.shape)# (10,2)(10,2)
    fakes = np.zeros((batch_size, 1))
    reals = np.ones((batch_size, 1))
    print(reals,'===',reals * 0.8)
    for epoch in range(epochs):

        noise = np.random.normal(0, 1, size=(batch_size, noiseSize))
        # print(noise.shape) (10,50)
        realImg = np.array(dataset.getRandomBatch(batch_size))
        fakeImg = model.generator.predict(noise)
        # print(realImg.shape, fakeImg.shape) (10,128,128,3) (10,128,128,3)
        model.discriminator.trainable = False

        noise = np.random.normal(0, 1, size=(batch_size, noiseSize))
        # print(noise.shape) (10,50)
        val = model.combine.train_on_batch(noise, reals*0.8)
        


        model.discriminator.trainable = True
        if random.randint(0,100) < 50 :
            trainX = np.concatenate([realImg, fakeImg])
            trainY = np.concatenate([reals, fakes])
            # print(trainX.shape, trainY.shape) (20,128,128,3)(20,2)
            # print(trainX, trainY)
        else:
            trainX = np.concatenate([fakeImg, realImg])
            trainY = np.concatenate([fakes, reals])
        #trainX, trainY = shuffle(trainXs, trainYs)
        if epoch % 5 == 0:
            dl = model.discriminator.train_on_batch(trainX, trainY)
        # print(dl) scalar training loss
        #d2 = model.discriminator.train_on_batch(realImg, reals)

        acc = accuracy(model.discriminator.predict(trainX), trainY)
        #dl = np.add(dl * 0.5, d2 * 0.5)
        print(epoch, "epoch G :", val, "D :", dl, 'd ACC :', acc)
        if epoch % 10 == 0:
            print(argmax(model.discriminator.predict(trainX), trainY))
            noise = np.random.normal(0, 1, size=(1, noiseSize))
            # print(noise.shape)(1, 100)
            img = model.generator.predict(noise)
            dataset.saveImgae(realImg[0], img[0], title + str(epoch))

        # if epoch % 100 == 0:
        #     for m in range(len(model.models)):
        #         dir = 'DATA\\Weights\\' + title + '\\' + saveOpt + '\\'
        #         if not os.path.exists(dir):
        #             os.makedirs(dir)
        #         model.models[m].save(dir + model.models[m].name + '.h5')
        #     print("Save Model")


train(100000)

