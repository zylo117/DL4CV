from cnn.nn.conv.microvggnet import MicroVGGNet
from cnn.nn.conv.alexnet import AlexNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = AlexNet.build(width=227, height=227, depth=3, classes=2)
plot_model(model, to_file="AlexNet.png", show_shapes=True)