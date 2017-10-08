import BinaryConverter as bc

images_filename = "Resources/t10k-images.idx3-ubyte"
labels_filename = "Resources/t10k-labels.idx1-ubyte"

def get_data_and_labels():
    print(">> Opening files...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print(">> Reading files...")
        images_file.read(4)
        
        num_of_items = int(images_file.read(4).encode('hex'), 16)
        num_of_rows = int(images_file.read(4).encode('hex'), 16)
        num_of_colums = int(images_file.read(4).encode('hex'), 16)
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)] for y in range(num_of_items)]
        labels = []

        for item in range(num_of_items):
            if item % 1000 == 0:
                print(">> Current image number: %7d" % item)
                
            for value in range(num_of_image_values):
                binaryValue = bc.convert(int(images_file.read(1).encode('hex'), 16))
                data[item][value] = binaryValue
            labels.append(int(labels_file.read(1).encode('hex'), 16))
        # print (data)
        return data, labels
    
    finally:
        images_file.close()
        labels_file.close()
        print(">> Files closed.")