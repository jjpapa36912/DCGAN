import tensorflow as tf
import tensorflow_addons as tfa

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class Model:
    
    def __init__(self, opts):
        self.noiseSize = opts['noiseSize']
        self.gf = 32
        self.df = 64

        self.generator = self.generatorModel('generator')
        # print(self.generator.output_shape[1])  <==64
        self.discriminator = self.discriminatorModel(
            self.generator.output_shape[1], 'discriminator')
        self.discriminator.summary()
        self.generator.summary()
        # self.discriminator.trainable = False
        inputs = tf.keras.layers.Input((self.noiseSize))
        genImage = self.generator(inputs)
        # print(genImage.shape)(None, 64, 64, 3)
        outputs = self.discriminator(genImage)

        self.combine = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name='combine')
        self.combine.summary()
        self.models = [self.discriminator, self.generator, self.combine]
        
    def generatorModel(self,name, kSize=(4,4)):
        inputs = tf.keras.layers.Input((self.noiseSize))
        print(inputs.shape)
        d = tf.keras.layers.Reshape((1,1,self.noiseSize))(inputs)
        d = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=kSize, strides=(4,4), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=kSize, strides=(2,2), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=kSize, strides=(2,2), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kSize, strides=(2,2), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kSize, strides=(2,2), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kSize, strides=(2,2), padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.ReLU()(d)
        d = tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kSize, strides=(2,2), padding='same', activation=tf.nn.tanh)(d)
       
        return tf.keras.models.Model(inputs=inputs, outputs=d, name=name)

    def discriminatorModel(self, inputSize, name, kSize=(4,4), strides=(2,2)):
        inputs = tf.keras.layers.Input((inputSize, inputSize, 3))
        d = tf.keras.layers.Conv2D(filters=128, kernel_size=kSize, strides=strides, padding='same')(inputs)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.LeakyReLU(0.2)(d)
        d = tf.keras.layers.Dropout(0.6)(d)
        d = tf.keras.layers.Conv2D(filters=256, kernel_size=kSize, strides=strides, padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.LeakyReLU(0.2)(d)
        d = tf.keras.layers.Dropout(0.6)(d)
        d = tf.keras.layers.Conv2D(filters=512, kernel_size=kSize, strides=strides, padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.LeakyReLU(0.2)(d)
        d = tf.keras.layers.Dropout(0.6)(d)
        d = tf.keras.layers.Conv2D(filters=1024, kernel_size=kSize, strides=strides, padding='same')(d)
        d = tfa.layers.InstanceNormalization()(d)
        d = tf.keras.layers.LeakyReLU(0.2)(d)
        d = tf.keras.layers.Dropout(0.6)(d)
        d = tf.keras.layers.Conv2D(filters=1, kernel_size=kSize, strides=(4,4), padding='same')(d)
        d = tf.keras.layers.Flatten()(d)
        d = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(d)

       
        return tf.keras.models.Model(inputs=inputs, outputs=d, name=name)

    

    def modelLoad(self):
        self.discriminator = self.models[0]
        self.generator = self.models[1]
        self.combine = self.models[2]

