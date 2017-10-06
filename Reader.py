import BinaryConverter as bc

images_filename = "Imagens/train-images.idx3-ubyte"
labels_filename = "Imagens/train-labels.idx1-ubyte"

def get_data_and_labels():
    print("Opening files ...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)] for y in range(num_of_items)]
        labels = []

        for item in range(num_of_items):
            # print("Current image number: %7d" % item)
            for value in range(num_of_image_values):
                binaryValue = bc.convert(int.from_bytes(images_file.read(1), byteorder="big"))
                data[item][value] = binaryValue
            labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
        # print (data)
        return data, labels
    
    finally:
        images_file.close()
        labels_file.close()
        print("Files closed.")