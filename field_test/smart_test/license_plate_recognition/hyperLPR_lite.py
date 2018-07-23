import cv2
import numpy as np
from keras import backend as K
from keras.models import *
from keras.layers import *

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济", "海", "民", "航", "空"
         ]


class LPR():
    def __init__(self, model_detection, model_finemapping, model_seq_rec):
        self.watch_cascade = cv2.CascadeClassifier(model_detection)
        self.modelFineMapping = self.model_finemapping()
        self.modelFineMapping.load_weights(model_finemapping)
        self.modelSeqRec = self.model_seq_rec(model_seq_rec)

    def computeSafeRegion(self, shape, bounding_rect):
        top = bounding_rect[1]  # y
        bottom = bounding_rect[1] + bounding_rect[3]  # y +  h
        left = bounding_rect[0]  # x
        right = bounding_rect[0] + bounding_rect[2]  # x +  w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left, top, right - left, bottom - top]

    def cropImage(self, image, rect):
        x, y, w, h = self.computeSafeRegion(image.shape, rect)
        return image[y:y + h, x:x + w]

    def detectPlateRough(self, image_gray, resize_h=720, en_scale=1.08, top_bottom_padding_rate=0.05):
        if top_bottom_padding_rate > 0.2:
            print(("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate))
            exit(1)
        height = image_gray.shape[0]
        padding = int(height * top_bottom_padding_rate)
        scale = image_gray.shape[1] / float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale * resize_h), resize_h))
        image_color_cropped = image[padding:resize_h - padding, 0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
        watches = self.watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),
                                                      maxSize=(36 * 40, 9 * 40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.15
            h += h * 0.3
            cropped = self.cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            cropped_images.append([cropped, [x, y + padding, w, h]])
        return cropped_images

    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(chars) + 1)
        res = table_pred.argmax(axis=1)
        for i, one in enumerate(res):
            if one < len(chars) and (i == 0 or (one != res[i - 1])):
                results += chars[one]
                confidence += table_pred[i][one]
        confidence /= len(results)
        return results, confidence

    def model_seq_rec(self, model_path):
        width, height, n_len, n_class = 164, 48, 7, len(chars) + 1
        rnn_size = 256
        input_tensor = Input((164, 48, 3))
        x = input_tensor
        base_conv = 32
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
            x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.25)(x)
        x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=input_tensor, outputs=x)
        base_model.load_weights(model_path)
        return base_model

    def model_finemapping(self):
        input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = Activation("relu", name='relu1')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = Activation("relu", name='relu2')(x)
        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = Activation("relu", name='relu3')(x)
        x = Flatten()(x)
        output = Dense(2, name="dense")(x)
        output = Activation("relu", name='relu4')(output)
        model = Model([input], [output])
        return model

    def finemappingVertical(self, image, rect):
        resized = cv2.resize(image, (66, 16))
        resized = resized.astype(np.float) / 255
        res_raw = self.modelFineMapping.predict(np.array([resized]))[0]
        res = res_raw * image.shape[1]
        res = res.astype(np.int)
        H, T = res
        H -= 3
        if H < 0:
            H = 0
        T += 2
        if T >= image.shape[1] - 1:
            T = image.shape[1] - 1
        rect[2] -= rect[2] * (1 - res_raw[1] + res_raw[0])
        rect[0] += res[0]
        image = image[:, H:T + 2]
        image = cv2.resize(image, (int(136), int(36)))
        return image, rect

    def recognizeOne(self, src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx, (164, 48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:, 2:, :]
        return self.fastdecode(y_pred)

    def SimpleRecognizePlateByE2E(self, image):
        images = self.detectPlateRough(image, image.shape[0], top_bottom_padding_rate=0.1)
        res_set = []
        for j, plate in enumerate(images):
            plate, rect = plate
            image_rgb, rect_refine = self.finemappingVertical(plate, rect)
            res, confidence = self.recognizeOne(image_rgb)
            res_set.append([res, confidence, rect_refine])
        return res_set
