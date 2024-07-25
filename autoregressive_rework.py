import sys
import os
import time
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
from tkinter.filedialog import askdirectory, askopenfilename


# Class to load in images from the dataset
class LoadImages:
    # Initialize the class
    def __init__(self, directory_of_images, file_with_labels, preprocessing):
        # Set the directory of images
        self.directory_of_images = directory_of_images
        # Set the preprocessing function
        self.preprocessing = preprocessing
        # Initialize the list of data locations
        self.data = []

        # Open the file with labels open as read
        with open(file_with_labels, "r") as current_file:
            # Read all lines in the file
            lines_in_file = current_file.readlines()
            # Loop through all lines in the file
            for current_line in lines_in_file:
                # Split the current line by the last underscore
                current_line = current_line.strip().rsplit("_", 0)
                # Add the current line to the list of image data locations
                self.data.append(current_line)
    
    # Get the length of the data list
    def __len__(self):
        # The amount of images found in the list
        return len(self.data)
    
    # Get the item at the index
    def __getitem__(self, idx):
        # Get the image location
        image_name = self.directory_of_images + "/" + self.data[idx][0]
        # Open the image with index idx
        curr_img = Image.open(image_name)\
        
        # Check if the image is grayscale and normalized (check that transform is applied)
        if self.preprocessing is not None:
            curr_img = self.preprocessing(curr_img)
        
        return curr_img

# Class to define the masked CNN
class convolutional_NN_msk(torch.nn.Conv2d):
	# Initialize the class
	def __init__(self, msk_type, *arguments, **keywordedarguments):
		# Set the mask type
		self.mask_type = msk_type
		torch.nn.Conv2d.__init__(self,*arguments, **keywordedarguments)
		self.register_buffer('buffer_msk', self.weight.data.clone())

		# Set the mask for the CNN
		_, _, h, w = self.weight.size()
		self.buffer_msk.fill_(1)
		if msk_type =='A':
			self.buffer_msk[:,:,h//2,w//2:] = 0
		else:
			self.buffer_msk[:,:,h//2,w//2+1:] = 0
		self.buffer_msk[:,:,h//2+1:,:] = 0

	# Propagation through the neural network
	def forward(self, input):
		self.weight.data*=self.buffer_msk
		return torch.nn.Conv2d.forward(self,input)

# Class to define the autoregressive CNN
class Model(torch.nn.Module):
	# Initialize the class
	def __init__(self, layer_numb=8, size_kernel = 7, ch_numb=64, dev=None):
		torch.nn.Module.__init__(self)
		# Set the amount of layers
		self.amount_of_layers = layer_numb
		# Set the kernel size
		self.kernel_size = size_kernel
		# Set the number of channels
		self.ch_numb = ch_numb
		# Set the device
		self.layer = {}
		# Set the device
		self.dev = dev

		# Set the first layer
		# A is so that the model does not have access to the future information the model is trying to predict
		self.Convolutional_first_layer = convolutional_NN_msk('A',1,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_first_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_first_layer = torch.nn.ReLU(True)

		# Set the second layer
		# B is so that the model does have access to the future information the model is trying to predict
		# Should help in learning more complex patterns
		self.Convolutional_second_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_second_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_second_layer = torch.nn.ReLU(True)

		# Set the third layer
		self.Convolutional_third_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_third_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_third_layer = torch.nn.ReLU(True)

		# Set the fourth layer
		self.Convolutional_fourth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_fourth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_fourth_layer = torch.nn.ReLU(True)

		# Set the fifth layer
		self.Convolutional_fifth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_fifth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_fifth_layer = torch.nn.ReLU(True)

		# Set the sixth layer
		self.Convolutional_sixth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_sixth_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_sixth_layer = torch.nn.ReLU(True)

		# Set the seventh layer
		self.Convolutional_seventh_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_seventh_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_seventh_layer = torch.nn.ReLU(True)

		# Set the eighth layer
		self.Convolutional_eight_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
		self.Normalization_eight_layer = torch.nn.BatchNorm2d(ch_numb)
		self.Activation_func_eight_layer = torch.nn.ReLU(True)

		if self.amount_of_layers == 10:
			# Set the ninth layer
			self.Convolutional_nineth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
			self.Normalization_nineth_layer = torch.nn.BatchNorm2d(ch_numb)
			self.Activation_func_nineth_layer = torch.nn.ReLU(True)

			# Set the tenth layer
			self.Convolutional_tenth_layer = convolutional_NN_msk('B',ch_numb,ch_numb, size_kernel, 1, size_kernel//2, bias=False)
			self.Normalization_tenth_layer = torch.nn.BatchNorm2d(ch_numb)
			self.Activation_func_tenth_layer = torch.nn.ReLU(True)

		# Set the output layer
		self.out = torch.nn.Conv2d(ch_numb, 256, 1)

	def forward(self, input):
		# Forward pass of all the layers
		input = self.Convolutional_first_layer(input)
		input = self.Normalization_first_layer(input)
		input = self.Activation_func_first_layer(input)

		input = self.Convolutional_second_layer(input)
		input = self.Normalization_second_layer(input)
		input = self.Activation_func_second_layer(input)

		input = self.Convolutional_third_layer(input)
		input = self.Normalization_third_layer(input)
		input = self.Activation_func_third_layer(input)

		input = self.Convolutional_fourth_layer(input)
		input = self.Normalization_fourth_layer(input)
		input = self.Activation_func_fourth_layer(input)

		input = self.Convolutional_fifth_layer(input)
		input = self.Normalization_fifth_layer(input)
		input = self.Activation_func_fifth_layer(input)

		input = self.Convolutional_sixth_layer(input)
		input = self.Normalization_sixth_layer(input)
		input = self.Activation_func_sixth_layer(input)

		input = self.Convolutional_seventh_layer(input)
		input = self.Normalization_seventh_layer(input)
		input = self.Activation_func_seventh_layer(input)

		input = self.Convolutional_eight_layer(input)
		input = self.Normalization_eight_layer(input)
		input = self.Activation_func_eight_layer(input)

		if self.amount_of_layers == 10:

			input = self.Convolutional_nineth_layer(input)
			input = self.Normalization_nineth_layer(input)
			input = self.Activation_func_nineth_layer(input)

			input = self.Convolutional_tenth_layer(input)
			input = self.Normalization_tenth_layer(input)
			input = self.Activation_func_tenth_layer(input)

		# Output layer
		return self.out(input)








def main():
	os.chdir("..")
	# Only use CUDA if wanted and available
	CUDA = True
	version = input("Which version of the model do you want to train? 1 = learning rate, 2 = Image size, 3 = Baseline")
	# path where the model is saved
	path_checkpoints = input("Where shoud the data (models and their generated images) be saved?")
	# Batch size of the model 
	batch_size = 128
	# No of layers in the model
	layers = 10
	# Kernel size of the model
	kernel = 7 # prev 7
	# Depth of the intermediate channels in the model
	channels = 400  #prev 400
	# No of epochs to train the model
	epochs = 50 
	# path where the NIH data is stored
	dataset_path = input("where are the NIH images stored (all together not in different folders)?") 
	# path where the training list is stored
	train_list = input("Where is the training list stored? (point exact to the file)")
	# dimensions for the images 
	img_dim = 64 # prev 128
	# Define the number of generated images to be saved
	gen_images = 128
	images_size = 64
	images_channels = 1

	if version == "2":
		img_dim_list = [128]
	else:
		img_dim_list = [64]
		
	for img_dim in img_dim_list:
		if img_dim == 32:
			layers = 8
			images_size = 32
			channels = 200
		elif img_dim == 128:
			layers = 10
			images_size = 128
			channels = 400
		else:
			layers = 10
			images_size = 64
			channels = 400

		# Define the transformation for the images (resize, grayscale, from PIL Image to tensor, normalize with mean 0,5 and standard deviation 0,5)
		transform_params = torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize((img_dim, img_dim)),
				torchvision.transforms.RandomHorizontalFlip(),
				torchvision.transforms.Grayscale(num_output_channels=1),
				torchvision.transforms.ToTensor(),
			]
		)
		# Load the images from the dataset
		dataset = LoadImages(str(dataset_path), str(train_list), transform_params)

		# Create the dataloader for the images
		train_load = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# Define the device
		if torch.cuda.is_available() and CUDA:
			dev = torch.device("cuda:0")
		else:
			dev = torch.device("cpu")
		print(dev)
		PixelCNN = Model().to(dev)
		
		# Using cross entropy loss
		cross_ent = torch.nn.CrossEntropyLoss()

		# Training the model
		print('Training Started')

		if version == "1":
			lr_list = [5e-4, 1e-4, 1e-5]
		else:
			lr_list = [1e-4]

		for lr in lr_list:
			# Define the optimizer, to update the weights
			Adam_optim = optim.Adam(PixelCNN.parameters(), lr=lr)
			# Loop through all the epochs
			for i in range(epochs):
				# activating learning mode
				PixelCNN.train(True)

				# Set the loss and step count
				step_count = 0
				loss_count = 0

				# Loop through all the batches
				for batch in train_load:
					
					# Set the batch and target image
					target_img = Variable(batch[:,0,:,:]*255).long()
					batch = batch.to(dev)
					target_img = target_img.to(dev)
					
					# Zero the gradients
					Adam_optim.zero_grad()

					# Propagation through the neural network
					output = PixelCNN(batch)

					# Calculate the loss
					loss_curr = cross_ent(output, target_img)
					# Backwardpropagation through the neural network to calculate the gradients
					loss_curr.backward()
					# Update the weights by performing a step
					Adam_optim.step()
					# Update the loss and step count
					loss_count+=loss_curr
					step_count+=1
					# Print the loss and step count
					print('Batch: '+str(step_count)+' Loss: '+str(loss_curr.item()))	

				print('Epoch: '+str(i)+' Over!')

				#Saving current PixelCNN state
				if not os.path.exists(path_checkpoints):
					os.makedirs(path_checkpoints)
				print("Saving Checkpoint!")
				if(i==epochs-1):
					torch.save(PixelCNN.state_dict(), str(path_checkpoints) +'/Model_Checkpoint_'+'Last'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+'.pt')
				else:
					torch.save(PixelCNN.state_dict(), str(path_checkpoints) +'/Model_Checkpoint_'+str(i)+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+'.pt')
				print('Checkpoint Saved')


				gen_img_tensor = torch.Tensor(gen_images, images_channels, images_size, images_size).to(dev)
				gen_img_tensor.fill_(0)

				#Generating images pixel by pixel
				for rows in range(images_size):
					for cols in range(images_size):
						out = PixelCNN(gen_img_tensor)
						probability = F.softmax(out[:,:,rows,cols], dim=-1).data
						gen_img_tensor[:,:,rows,cols] = torch.multinomial(probability, 1)
						gen_img_tensor[:,:,rows,cols] = gen_img_tensor[:,:,rows,cols].float()
						gen_img_tensor[:,:,rows,cols] = gen_img_tensor[:,:,rows,cols] / 255.0

				#Saving images row wise
				torchvision.utils.save_image(gen_img_tensor, str(path_checkpoints) + 'sample' + str(epochs)+ '.png', nrow=12, padding=0)


	print('Training Finished!')


if __name__=="__main__":
	main()



