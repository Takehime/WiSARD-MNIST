from PyWANN import WiSARD as wi
import numpy as np
import Reader as r

#  C:\Python27\python.exe Main.py "Resources/train-images.idx3-ubyte" "Resources/train-labels.idx1-ubyte" "Resources/t10k-images.idx3-ubyte" "Resources/t10k-labels.idx1-ubyte"

def train_and_predict(train_images_filename, train_labels_filename, 
    pred_images_filename, pred_labels_filename, 
    num_of_bits, bleaching_value):

    train_data, train_labels = r.get_data_and_labels(train_images_filename, train_labels_filename, 
        images_to_read = 5000, print_progress = False)

    # retina_length = 784
    num_bits_addr = num_of_bits
    bleaching = True

    w = wi.WiSARD(num_bits_addr, bleaching, defaul_b_bleaching = bleaching_value)

    # print(">> Starting training...")
    w.fit(train_data, train_labels)

    pred_data, pred_labels = r.get_data_and_labels(pred_images_filename, pred_labels_filename, 
    images_to_read = 5000, print_progress = False)

    # print(">> Starting prediction...")

    result_labels = []
    for i in range(len(pred_labels)):
        # if i % 1000 == 0:
            # print(">> Current image number: %7d" % i)
        result_labels.append(w.predict([pred_data[i]])[0])
  
    # result_labels = w.predict(pred_data)

    successes = 0
    for i in xrange(len(pred_labels)):
        if (pred_labels[i] == result_labels[i]):
            successes += 1

    accuracy = (float(successes) * 100) / len(pred_labels)
    print(">> num_bits_addr: " + str(num_bits_addr) + "; Images trained: " + str(len(train_data)) 
        + "; Images predicted: " + str(len(pred_data)) + "; Bleaching value: " + str(bleaching_value))
    print(">> Accuracy: " + str(accuracy) + "%")

