from tensorflow import keras
from tensorflow.keras import layers


class ResNetModel:
    def __init__(self, input_shape=(512, 512, 3), classes=2):
        self.input_shape = input_shape
        self.classes = classes

    # 恒等模块——identity_block
    def identity_block(self, X, f, filters, stage, block):
        """
        三层的恒等残差块
        param :
            X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
            f -- 整数，指定主路径的中间 CONV 窗口的形状
            filters -- python整数列表，定义主路径的CONV层中的过滤器数目
            stage -- 整数，用于命名层，取决于它们在网络中的位置
            block --字符串/字符，用于命名层，取决于它们在网络中的位置
        return:
            X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)
        """
        # 定义基本的名字
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        # 过滤器
        F1, F2, F3 = filters
        # 保存输入值，后将输入值返回主路径
        X_shortcut = X

        # 主路径第一部分
        X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                          name=conv_name_base + "2a", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = layers.Activation("relu")(X)

        # 主路径第二部分
        X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                          name=conv_name_base + "2b", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        X = layers.Activation("relu")(X)

        # 主路径第三部分
        X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                          name=conv_name_base + "2c", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

        # 主路径最后部分,为主路径添加shortcut并通过relu激活
        X = layers.Add()([X, X_shortcut])
        X = layers.Activation("relu")(X)
        return X

    # 卷积残差块——convolutional_block
    def convolutional_block(self, X, f, filters, stage, block, s=2):
        """
        param :
        X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
        f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
        filters -- python整数列表，定义主路径的CONV层中过滤器的数目
        stage -- 整数，用于命名层，取决于它们在网络中的位置
        block --字符串/字符，用于命名层，取决于它们在网络中的位置
        s -- 整数，指定使用的步幅
        return:
        X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
        """
        # 定义基本名字
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"
        # 过滤器
        F1, F2, F3 = filters
        # 保存输入值，后将输入值返回主路径
        X_shortcut = X

        # 主路径第一部分
        X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
                          name=conv_name_base + "2a", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = layers.Activation("relu")(X)

        # 主路径第二部分
        X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                          name=conv_name_base + "2b", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        X = layers.Activation("relu")(X)

        # 主路径第三部分
        X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                          name=conv_name_base + "2c", kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

        # shortcut路径
        X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                                   name=conv_name_base + "1",
                                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = layers.BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

        # 主路径最后部分,为主路径添加shortcut并通过relu激活
        X = layers.Add()([X, X_shortcut])
        X = layers.Activation("relu")(X)

        return X

    # 50层ResNet模型构建
    def ResNet50(self):
        """
        构建50层的ResNet,结构为：
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        param :
            input_shape -- 数据集图片的维度
            classes -- 整数，分类的数目
        return:
            model -- Keras中的模型实例
        """
        # 将输入定义为维度大小为 input_shape的张量
        X_input = layers.Input(self.input_shape)
        # Zero-Padding
        X = layers.ZeroPadding2D((3, 3))(X_input)
        # Stage 1
        X = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
                          kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = layers.BatchNormalization(axis=3, name="bn_conv1")(X)
        X = layers.Activation("relu")(X)
        X = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)
        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
        X = self.identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
        X = self.identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")
        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
        for i in range(7):
            X = self.identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b" + str(i))
        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
        for i in range(35):
            X = self.identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b" + str(i))
        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
        X = self.identity_block(X, f=3, filters=[256, 256, 2048], stage=5, block="b")
        X = self.identity_block(X, f=3, filters=[256, 256, 2048], stage=5, block="c")
        # 最后阶段
        # 平均池化
        X = layers.AveragePooling2D(pool_size=(2, 2))(X)
        # 输出层
        X = layers.Flatten()(X)
        # 展平
        X = layers.Dense(self.classes, activation="softmax", name="fc" + str(self.classes),
                         kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        # 创建模型
        model = keras.models.Model(inputs=X_input, outputs=X, name="ResNet50")
        return model

    def res_net_block(self, input_data, filters, conv_size):
        # CNN层
        x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
        # 第二层没有激活函数
        x = layers.BatchNormalization()(x)
        # 两个张量相加
        x = layers.Add()([x, input_data])
        # 对相加的结果使用ReLU激活
        x = layers.Activation('relu')(x)
        # 返回结果
        return x

    def ResNet(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D(3)(x)
        num_res_net_blocks = 5
        for i in range(num_res_net_blocks):
            x = self.res_net_block(x, 64, 3)
        # 添加一个CNN层
        x = layers.Conv2D(64, 3, activation='relu')(x)
        # 全局平均池化GAP层
        x = layers.GlobalAveragePooling2D()(x)
        # 几个密集分类层
        x = layers.Dense(256, activation='relu')(x)
        # 退出层
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        res_net_model = keras.Model(inputs, outputs)

        return res_net_model
