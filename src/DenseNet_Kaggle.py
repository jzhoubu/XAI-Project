from keras.layers import BatchNormalization,Conv2D,Input,Concatenate,Dense,Dropout,MaxPool2D,AveragePooling2D,Activation,GlobalAveragePooling2D,Reshape,multiply
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l1_l2
from keras.models import Model,load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def build_model(shape, target_size,k, kernel_initializer="he_normal"):
    def sub_dense_block(x, k, kernel_initializer="he_normal"):
        conv1 = Conv2D(k, padding="same", kernel_size=(1, 1), kernel_initializer=kernel_initializer,
                        kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
        conv2 = Conv2D(k, padding="same", kernel_size=(7, 7),use_bias=False, kernel_initializer=kernel_initializer)

        with tf.name_scope("sub_dense_block"):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = conv1(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = conv2(x)
        return x

    def dense_block(x, k, block_size, kernel_initializer="he_normal"):
        output_set = [x]
        yi = None
        with tf.name_scope("dense_block"):
            for i in range(block_size):
                if len(output_set) == 1:
                    xi = output_set[0]
                else:
                    xi = Concatenate()(output_set)
                yi = sub_dense_block(xi, k, kernel_initializer=kernel_initializer)
                output_set.append(yi)
        return yi
    def se_block(x,out_dim):
        squeeze = GlobalAveragePooling2D()(x)
        excitation = Dense(units=out_dim // 4,activation='relu')(squeeze)
        excitation = Dense(units=out_dim,activation='sigmoid')(excitation)
        excitation = Reshape((1,1,out_dim))(excitation)
        scale = multiply([x,excitation])
        return scale
    def transition_layer(x, k, kernel_initializer="he_normal"):
        conv1 = Conv2D(k, padding="same", kernel_size=(1, 1),use_bias=False, kernel_initializer=kernel_initializer)
        with tf.name_scope("transition_layer"):
            x = conv1(x)
            x = se_block(x,k)
            x = AveragePooling2D()(x)
        return x

    inp = Input(shape=shape)
    x = Conv2D(k, kernel_size=(3, 3), padding='same',use_bias=False, kernel_initializer=kernel_initializer)(inp)
    x = AveragePooling2D()(x)
    # dense block 1
    x = dense_block(x, k, 3, kernel_initializer)
    x = transition_layer(x, k, kernel_initializer)
    #dense block 2
    x = dense_block(x, k, 3, kernel_initializer)
    x = transition_layer(x, k, kernel_initializer)
     # dense block 3
    x = dense_block(x, k, 3, kernel_initializer)
    x = transition_layer(x, k, kernel_initializer)
    # dense block 4
    x = dense_block(x, k, 3, kernel_initializer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2*k, activation="relu",kernel_initializer=kernel_initializer)(x)
    y = Dense(units=target_size, activation="softmax",kernel_initializer=kernel_initializer)(x)
    return Model(inp, y)