import Classifier as c
import sys

# train_images_filename = "Resources/train-images.idx3-ubyte"
# train_labels_filename = "Resources/train-labels.idx1-ubyte"
# pred_images_filename = "Resources/t10k-images.idx3-ubyte"
# pred_labels_filename = "Resources/t10k-labels.idx1-ubyte"

if len(sys.argv) != 5:
    print ("Wrong number os arguments! Needed: 5, Written: " + len(sys.argv))
    print ("program_name -train_images -train_labels -predict_images -predict_labels")
    sys.exit(0)
else:
    train_images_filename = sys.argv[1]
    train_labels_filename = sys.argv[2]

    pred_images_filename = sys.argv[3]
    pred_labels_filename = sys.argv[4]

    for confidence in range(5, 15):
        confidence = confidence/float(10)
        c.train_and_predict(
            train_images_filename, 
            train_labels_filename, 
            pred_images_filename, 
            pred_labels_filename, 
            readable_images_train = 5000,
            readable_images_predict = 5000,            
            confidence_value = confidence)