# WiSARD-MNIST
> (ASSIGNMENT) Use WiSARD neural network to identify handwritten characters from MNIST database.

## requirements
- Python 2.7
- PyWANN library: https://github.com/firmino/PyWANN
- MNIST datasets are included (_Resources_ folder).

## how to run
> Classifier.py [train_image_file] [train_label_file] [test-image-file] [test-label-file] [options]

**options:**
- --num_bits_addr INT: sets number of address bits (vanilla WiSARD parameter)
- --bleaching BOOL: turns bleaching on and off (default: True)
- --bleaching_initial_value INT: initial value for bleaching threshold (default: 1)
- --bleaching_confidence FLOAT: confidence value used in bleaching ties (default: 0.1)

## info
- Made by: Silvia Pimpão, Vinícius Garcia
- For: Redes Neurais Sem Peso (2017/2)
