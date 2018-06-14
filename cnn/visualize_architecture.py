from cnn.nn.conv.microvggnet import MicroVGGNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = MicroVGGNet.build(width=224, height=224, depth=1, classes=20)
plot_model(model, to_file="MicroVGG.png", show_shapes=True)