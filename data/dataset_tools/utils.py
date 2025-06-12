import os
import random

def separate_elements(file_path):
    list1 = []
    list2 = []

    with open(file_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 2:
                list1.append(elements[0])
                list2.append(elements[1])

    return list1, list2

def create_path(IMGS_PATH, list_new_files):
    '''
    Util function to add the file path of all the images to the list of names of the selected 
    images that will form the valid ones.
    '''
    file_path, name = os.path.split(
        IMGS_PATH[0])  # we pick only one element of the list
    output = [os.path.join(file_path, element) for element in list_new_files]

    return output

def flatten_list_comprehension(matrix):
    return [item for row in matrix for item in row]

def check_paths(list_of_lists):
    '''
    check if all the image routes are correct
    '''
    paths = flatten_list_comprehension(list_of_lists)
    trues = [(os.path.isfile(file), file) for file in paths]
    for true,file in trues:
        if true != True:
            print('Non valid route!')
            print(file)



