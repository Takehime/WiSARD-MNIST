from PyWANN import WiSARD as wi
import numpy as np
import Reader as r

# 1st read: train
train_images_filename = "Resources/train-images.idx3-ubyte"
train_labels_filename = "Resources/train-labels.idx1-ubyte"
train_data, train_labels = r.get_data_and_labels(train_images_filename, train_labels_filename, print_progress = True)

# retina_length = 784
num_bits_addr = 16
bleaching = True

w = wi.WiSARD(num_bits_addr, bleaching)

# training discriminators
print(">> Starting training...")
w.fit(train_data, train_labels)

# 2nd read: predict
pred_images_filename = "Resources/t10k-images.idx3-ubyte"
pred_labels_filename = "Resources/t10k-labels.idx1-ubyte"
pred_data, pred_labels = r.get_data_and_labels(pred_images_filename, pred_labels_filename)

print(">> Starting prediction...")
# array = np.zeros((1, 784))
# print (str(pred_data))

# pred_labels = train_labels
# pred_data = train_data
result_labels = []
for i in range(len(pred_labels)):
    if i % 1000 == 0:
        print(">> Current image number: %7d" % i)
    result_labels.append(w.predict([pred_data[i]])[0])

# result_labels = w.predict(pred_data)

successes = 0
for i in xrange(len(pred_labels)):
    if (pred_labels[i] == result_labels[i]):
        successes += 1

accuracy = (float(successes) * 100) / len(pred_labels)
print(">> num_bits_addr: " + str(num_bits_addr) + "; Images trained: " + str(len(train_data)) + "; Images predicted: " + str(len(pred_data)))
print(">> Accuracy: " + str(accuracy) + "%")