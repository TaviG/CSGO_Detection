import sys
import read

path = sys.argv[1]
train = read.get_path('train')
valid = read.get_path('valid')
test =  read.get_path('test')

train_images, train_labels = read.get_files( path + train )
valid_images, valid_labels = read.get_files( path + valid )
test_images, test_labels = read.get_files( path + test )

read.display_images(train_images, 3)
print(test_labels)