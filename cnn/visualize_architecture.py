from cnn.nn.conv.microvggnet import MicroVGGNet
from cnn.nn.conv.alexnet import AlexNet
from keras.applications import InceptionResNetV2
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
# lpr_model = AlexNet.build(width=227, height=227, depth=3, classes=2)
model = InceptionResNetV2(weights='imagenet',input_shape=(299, 299, 3), include_top=False)
plot_model(model, to_file="InceptionResNetV2_notop.png", show_shapes=True)