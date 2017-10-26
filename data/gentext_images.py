import os
import os.path

# train_labels_file = open('train_labels.txt', 'w')
train_images_file = open('train_images.txt', 'w')
val_images_file  = open('val_images.txt', 'w')
# val_labels_file   = open('val_labels.txt', 'w')
test_images_file  = open('test_images.txt', 'w')




for dirpath, dirnames, filenames in os.walk("."):
	for filename in [f for f in filenames if f.endswith(".png")]:
		checkpath = os.path.split(os.path.split(dirpath)[0])[1]
		
		if checkpath == 'val':
			path = os.path.join(dirpath, filename)
			val_images_file.writelines(path[2:] +"\n")  # python will convert \n to os.linesep
						
		if checkpath == 'train':
			path = os.path.join(dirpath, filename)
			train_images_file.writelines(path[2:] +"\n")  # python will convert \n to os.linesep

		if checkpath == 'test':
			path = os.path.join(dirpath, filename)
			test_images_file.writelines(path[2:] +"\n")  # python will convert \n to os.linesep			

train_images_file.close()
val_images_file.close()
test_images_file.close()  	

