import os
import os.path

train_labels_file = open('train_labels.txt', 'w')
# train_images_file = open('train_images.txt', 'w')
#val_images_file  = open('val_images.txt', 'w')
val_labels_file   = open('val_labels.txt', 'w')
#test_images_file  = open('test_images.txt', 'w')




for dirpath, dirnames, filenames in os.walk("."):
	for filename in [f for f in filenames if f.endswith("_labelIds.png")]:
		checkpath = os.path.split(os.path.split(dirpath)[0])[1]

		if checkpath == 'val':
			path = os.path.join(dirpath, filename)
			val_labels_file.writelines(path[2:] +"\n")

		if checkpath == 'train':
			path = os.path.join(dirpath, filename)
			train_labels_file.writelines(path[2:] +"\n")		

train_labels_file.close()  	
val_labels_file.close()


