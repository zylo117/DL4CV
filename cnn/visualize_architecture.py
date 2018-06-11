from starter.cnn.nn.conv.lenet import LeNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(width=28, height=28, depth=1, classes=10)
plot_model(model, to_file="lenet.png", show_shapes=True)