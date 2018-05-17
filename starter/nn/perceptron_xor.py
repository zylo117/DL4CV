from starter.nn.perception import Perception
import numpy as np

# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define out perception and train it
print("[INFO] training perception...")
p = Perception(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# now that our perception is trained we can evaluate it
print("[INFO] testing perception...")

# new that our network is trained, loop over the data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to our console
    pred = p.predict(x)
    print("[INFO] data=%s, ground-truth=%s, pred=%s" % (x, target[0], pred))
