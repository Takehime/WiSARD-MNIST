import BinaryConverter as bc

def get_data_and_labels(
    images_filename,
    labels_filename,
    images_to_read = -1,
    print_progress = False):

    if print_progress:
        print(">> Opening files...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        if print_progress:
            print(">> Reading files...")

        images_file.read(4)
        
        if images_to_read == -1:
            num_of_items = int(images_file.read(4).encode('hex'), 16)
        else:
            images_file.read(4)
            num_of_items = images_to_read

        num_of_rows = int(images_file.read(4).encode('hex'), 16)
        num_of_colums = int(images_file.read(4).encode('hex'), 16)
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = [[None for x in range(num_of_image_values)] for y in range(num_of_items)]
        labels = []

        for item in range(num_of_items):
            if item % 100 == 0 and print_progress:
                print(">> Current image number: %7d" % item)
                
            aux = []
            for value in range(num_of_image_values):
                aux.append(int(images_file.read(1).encode('hex'), 16))

            binary_image = bc.convert(aux)
            
            for value in range(num_of_image_values):
                data[item][value] = binary_image[value]

            labels.append(int(labels_file.read(1).encode('hex'), 16))
        # print (data)
        return data, labels
    
    finally:
        images_file.close()
        labels_file.close()
        if print_progress:
            print(">> Files closed.")