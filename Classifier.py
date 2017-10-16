from PyWANN import WiSARD as wi
import numpy as np
import Reader as r
import sys

#  C:\Python27\python.exe Main.py "Resources/train-images.idx3-ubyte" "Resources/train-labels.idx1-ubyte" "Resources/t10k-images.idx3-ubyte" "Resources/t10k-labels.idx1-ubyte"

def train_and_predict(
    train_images_filename, 
    train_labels_filename, 
    pred_images_filename, 
    pred_labels_filename, 
    num_of_bits = 27, 
    bleaching_value = 1, 
    confidence_value = 0.1,
    readable_images_train = -1,
    readable_images_predict = -1,
    bleaching = True,
    print_progress = False):

    train_data, train_labels = r.get_data_and_labels(
        train_images_filename, 
        train_labels_filename, 
        images_to_read = readable_images_train, 
        print_progress = print_progress)

    # retina_length = 784
    w = wi.WiSARD(
        num_of_bits, 
        bleaching = bleaching, 
        defaul_b_bleaching = bleaching_value, 
        confidence_threshold = confidence_value)

    if print_progress:
        print(">> Starting training...")
    w.fit(train_data, train_labels)

    pred_data, pred_labels = r.get_data_and_labels(
        pred_images_filename, 
        pred_labels_filename, 
        images_to_read = readable_images_predict, 
        print_progress = print_progress)
    
    if print_progress:
        print(">> Starting prediction...")

    result_labels = []
    for i in range(len(pred_labels)):
        if print_progress:
            if i % 1000 == 0:
                print(">> Current image number: %7d" % i)
        result_labels.append(w.predict([pred_data[i]])[0])
  
    # result_labels = w.predict(pred_data)

    successes = 0
    for i in xrange(len(pred_labels)):
        if (pred_labels[i] == result_labels[i]):
            successes += 1

    accuracy = (float(successes) * 100) / len(pred_labels)
    print(">> num_bits_addr: " + str(num_of_bits) + "; Images trained: " + str(len(train_data)) 
        + "; Images predicted: " + str(len(pred_data)) + "; Bleaching value: " + str(bleaching_value)
        + "; Confidence threshold: " + str(confidence_value))
    print(">> Accuracy: " + str(accuracy) + "%")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print ("Wrong number os arguments! Needed: 5, Written: " + len(sys.argv))
        print ("program_name -train_images -train_labels -predict_images -predict_labels")
        sys.exit(0)
    else:
        train_images_filename = sys.argv[1]
        train_labels_filename = sys.argv[2]

        pred_images_filename = sys.argv[3]
        pred_labels_filename = sys.argv[4]

        num_bits_addr = 27
        bleaching = True
        confidence_value = 0.1
        initial_bleaching_value = 1
        print_progress = False

        #exclude obligatory parameters such as test/train files and script name
        parameters = list(sys.argv)[5:]

        for i in range(len(parameters) - 1, -1, -1):
            if parameters[i] == "--print_progress":
                if len(parameters) <= i + 1:
                    print ">> Error: no parameter for flag 'print_progress'"
                    sys.exit(0)
                print_progress = (parameters[i + 1] == "True")
                parameters.pop()
                parameters.pop()
            elif parameters[i] == "--bleaching":
                if len(parameters) <= i + 1:
                    print ">> Error: no parameter for flag 'bleaching'"
                    sys.exit(0)
                bleaching = (parameters[i + 1] == "True")
                parameters.pop()
                parameters.pop()
            elif parameters[i] == "--bleaching_initial_value":
                if len(parameters) <= i + 1:
                    print ">> Error: no parameter for flag 'bleaching_initial_value'"
                    sys.exit(0)
                initial_bleaching_value = (int(parameters[i + 1]))
                parameters.pop()
                parameters.pop()
            elif parameters[i] == "--bleaching_confidence":
                if len(parameters) <= i + 1:
                    print ">> Error: no parameter for flag 'bleaching_confidence'"
                    sys.exit(0)
                confidence_value = (float(parameters[i + 1]))
                parameters.pop()
                parameters.pop()
            elif parameters[i] == "--num_bits_addr":
                if len(parameters) <= i + 1:
                    print ">> Error: no parameter for flag 'num_bits_addr'"
                    sys.exit(0)
                num_bits_addr = (int(parameters[i + 1]))
                parameters.pop()
                parameters.pop()
            elif parameters[i].startswith("--"):
                print ">> Error: could not understand flag '" + parameters[i] + "'!"
                sys.exit(0)

        if len(parameters) > 0:
            print ">> Error: could not understand parameter '" + parameters[len(parameters) - 1] + "'!"

        train_and_predict(
            train_images_filename, 
            train_labels_filename, 
            pred_images_filename, 
            pred_labels_filename,
            bleaching = bleaching,
            num_of_bits = num_bits_addr,
            bleaching_value = initial_bleaching_value,            
            confidence_value = confidence_value)