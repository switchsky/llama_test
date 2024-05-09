import os

def find_max_number_file(directory):
    max_number = 0
    for filename in os.listdir(directory):
        if filename.endswith('.png') and filename[:-4].isdigit():
            number = int(filename[:-4])
            max_number = max(max_number, number)
    return max_number