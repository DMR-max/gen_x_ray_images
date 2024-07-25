import sys
import os
import time
import torch
import torchvision
from torch import optim
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
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
        # Return the number of images in the dataset
        return len(self.data)
    
    # Get the item at the index
    def __getitem__(self, idx):
        # Get the image location
        image_name = self.directory_of_images + "/" + self.data[idx][0]
        # Open the image with index idx
        curr_image = Image.open(image_name)\
        
        # Check if the image is grayscale and normalized (check that transform is applied)
        if self.preprocessing is not None:
            curr_image = self.preprocessing(curr_image)
        
        return curr_image

# Class to load in images from the dataset
class MaskedCNN(nn.Conv2d):
	# Initialize the class
	def __init__(self, Type_of_mask, *arguments, **keywordedarguments):
		# Set the mask type
		self.mask_type = Type_of_mask
		nn.Conv2d.__init__(self,*arguments, **keywordedarguments)
		self.register_buffer('buffer_mask', self.weight.data.clone())

		# Set the mask for the CNN
		_, _, h, w = self.weight.size()
		self.buffer_mask.fill_(1)
		if Type_of_mask =='A':
			self.buffer_mask[:,:,h//2,w//2:] = 0
			self.buffer_mask[:,:,h//2+1:,:] = 0
		else:
			self.buffer_mask[:,:,h//2,w//2+1:] = 0
			self.buffer_mask[:,:,h//2+1:,:] = 0

	# Forward pass
	def forward(self, input):
		self.weight.data*=self.buffer_mask
		return nn.Conv2d.forward(self,input)

# Class to define the model
class Model(nn.Module):
	# Initialize the class
	def __init__(self, amount_of_layers=8, kernel_size = 7, ch_numb=64, dev=None):
		nn.Module.__init__(self)
		# Set the amount of layers
		self.amount_of_layers = amount_of_layers
		# Set the kernel size
		self.kernel_size = kernel_size
		# Set the number of channels
		self.ch_numb = ch_numb
		# Set the device
		self.layer = {}
		# Set the device
		self.dev = dev

		# Set the first layer
		# A is so that the model does not have access to the future information the model is trying to predict
		self.Convolutional_layer_1 = MaskedCNN('A',1,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_1 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_1= nn.ReLU(True)

		# Set the second layer
		# B is so that the model does have access to the future information the model is trying to predict
		# Should help in learning more complex patterns
		self.Convolutional_layer_2 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_2 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_2= nn.ReLU(True)

		# Set the third layer
		self.Convolutional_layer_3 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_3 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_3= nn.ReLU(True)

		# Set the fourth layer
		self.Convolutional_layer_4 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_4 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_4= nn.ReLU(True)

		# Set the fifth layer
		self.Convolutional_layer_5 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_5 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_5= nn.ReLU(True)

		# Set the sixth layer
		self.Convolutional_layer_6 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_6 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_6= nn.ReLU(True)

		# Set the seventh layer
		self.Convolutional_layer_7 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_7 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_7= nn.ReLU(True)

		# Set the eighth layer
		self.Convolutional_layer_8 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
		self.Normalization_8 = nn.BatchNorm2d(ch_numb)
		self.Activation_func_8= nn.ReLU(True)

		if self.amount_of_layers == 10:
			# Set the ninth layer
			self.Convolutional_layer_9 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
			self.Normalization_9 = nn.BatchNorm2d(ch_numb)
			self.Activation_func_9= nn.ReLU(True)

			# Set the tenth layer
			self.Convolutional_layer_10 = MaskedCNN('B',ch_numb,ch_numb, kernel_size, 1, kernel_size//2, bias=False)
			self.Normalization_10 = nn.BatchNorm2d(ch_numb)
			self.Activation_func_10= nn.ReLU(True)

		# Set the output layer
		self.out = nn.Conv2d(ch_numb, 256, 1)

	def forward(self, x):
		# Forward pass of all the layers
		x = self.Convolutional_layer_1(x)
		x = self.Normalization_1(x)
		x = self.Activation_func_1(x)

		x = self.Convolutional_layer_2(x)
		x = self.Normalization_2(x)
		x = self.Activation_func_2(x)

		x = self.Convolutional_layer_3(x)
		x = self.Normalization_3(x)
		x = self.Activation_func_3(x)

		x = self.Convolutional_layer_4(x)
		x = self.Normalization_4(x)
		x = self.Activation_func_4(x)

		x = self.Convolutional_layer_5(x)
		x = self.Normalization_5(x)
		x = self.Activation_func_5(x)

		x = self.Convolutional_layer_6(x)
		x = self.Normalization_6(x)
		x = self.Activation_func_6(x)

		x = self.Convolutional_layer_7(x)
		x = self.Normalization_7(x)
		x = self.Activation_func_7(x)

		x = self.Convolutional_layer_8(x)
		x = self.Normalization_8(x)
		x = self.Activation_func_8(x)

		if self.amount_of_layers == 10:

			x = self.Convolutional_layer_9(x)
			x = self.Normalization_9(x)
			x = self.Activation_func_9(x)

			x = self.Convolutional_layer_10(x)
			x = self.Normalization_10(x)
			x = self.Activation_func_10(x)

		# Output layer
		return self.out(x)








def main():
	version = input("Which version of the model do you want to train? 1 = learning rate, 2 = Image size, 3 = Baseline")
	print("Where shoud the data (models and their generated images) be saved?")
	# path where the model is saved
	save_path = askdirectory()
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
	print("where are the NIH images stored (all together not in different folders)?")
	# path where the NIH data is stored
	dataset_path = askdirectory() 
	print("Where is the training list stored?")
	# path where the training list is stored
	train_list = askopenfilename()
	# dimensions for the images 
	img_dim = 64 # prev 128
	# Define the number of generated images to be saved
	no_images = 128
	images_size = 64
	images_channels = 1

	if version == "2":
		img_dim_list = [32, 64, 128]
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
		transform_params = transforms.Compose(
			[
				transforms.Resize((img_dim, img_dim)),
				transforms.RandomHorizontalFlip(),
				transforms.Grayscale(num_output_channels=1),
				transforms.ToTensor(),
			]
		)
		# Load the images from the dataset
		dataset = LoadImages(str(dataset_path), str(train_list), transform_params)

		# Create the dataloader for the images
		Dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# Define the device
		dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print(dev)
		PixelCNN_model = Model().to(dev)
		PixelCNN_model.load_state_dict(torch.load('/data/s3287297/models_baseline_try_3/Model_Checkpoint_Last.pt'))
		
		# Define the loss function
		cross_ent_loss = nn.CrossEntropyLoss()

		# Training the model
		Total_loss = []
		print('Training Started')

		if version == "1":
			lr_list = [5e-4, 1e-4, 1e-5]
		else:
			lr_list = [1e-4]

		for lr in lr_list:
			# Define the optimizer, to update the weights
			Adam_optim = optim.Adam(PixelCNN_model.parameters(), lr=lr)
			# Training the model
			for i in range(epochs):
				# Set the model to training mode
				PixelCNN_model.train(True)

				# Initialize the loss and step count
				step_count = 0
				loss_count = 0

				# Loop through all the batches
				for batch in Dataloader:
					
					# Set the batch and target image
					target_img = Variable(batch[:,0,:,:]*255).long()
					batch = batch.to(dev)
					target_img = target_img.to(dev)
					
					# Zero the gradients
					Adam_optim.zero_grad()

					# Forward pass
					output = PixelCNN_model(batch)

					# Calculate the loss
					loss_curr = cross_ent_loss(output, target_img)
					# Backward pass to update the weights
					loss_curr.backward()
					# Update the weights by performing a step
					Adam_optim.step()
					# Update the loss and step count
					loss_count+=loss_curr
					step_count+=1
					# Print the loss and step count
					print('Batch: '+str(step_count)+' Loss: '+str(loss_curr.item()))	

				print('Epoch: '+str(i)+' Over!')

				#Saving the model
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				print("Saving Checkpoint!")
				if(i==epochs-1):
					torch.save(PixelCNN_model.state_dict(), save_path+'/Model_Checkpoint_'+'Last'+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+'.pt')
				else:
					torch.save(PixelCNN_model.state_dict(), save_path+'/Model_Checkpoint_'+str(i)+ '_lr='+str(lr)+ 'img_dim='+str(img_dim)+'.pt')
				print('Checkpoint Saved')

				# model = config['model']
				# images = config['images']

				# load_path = model.get('load_path', 'Models/Model_Checkpoint_Last.pt')
				# assert os.path.exists(load_path), 'Saved Model File Does not exist!'
				# no_images = images.get('no_images', 128)
				# images_size = images.get('images_size', 64)
				# images_channels = images.get('images_channels', 1)
				

				# #Define and load model
				# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
				# net = PixelCNN().to(device)
				# if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
				# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
				# 	net = nn.DataParallel(net)
				# net.load_state_dict(torch.load(load_path))
				# net.eval()



				sample = torch.Tensor(no_images, images_channels, images_size, images_size).to(dev)
				sample.fill_(0)

				#Generating images pixel by pixel
				for i in range(images_size):
					for j in range(images_size):
						out = PixelCNN_model(sample)
						probs = F.softmax(out[:,:,i,j], dim=-1).data
						sample[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

				#Saving images row wise
				torchvision.utils.save_image(sample, 'sample' + str(epochs)+ '.png', nrow=12, padding=0)


	print('Training Finished!')


if __name__=="__main__":
	main()



