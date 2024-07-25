import os
import sys
import math
import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import configparser

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, txt_file, trans=None):
        self.folder = folder
        self.trans = trans
        self.image_names = []
        self.target = []

        # Read the txt file containing information about dataset split and possibly labels
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                target, _ = line.strip().rsplit("_",1)  # Assuming each line contains image_name and target label
                image_name = line.strip()
                self.image_names.append(image_name)
                self.target.append(int(target))  # Convert target to int

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.image_names[idx])
        image = Image.open(img_name)

        if self.trans:
            image = self.trans(image)

        target = self.target[idx]
        return image, target  
    

class Discriminator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, D_HIDDEN, img_size):
        super(Discriminator, self).__init__()
        if img_size == 64:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(D_HIDDEN * 16),
                # nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif img_size == 128:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 4th layer
                nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(D_HIDDEN * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 2nd layer
                nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # # 3rd layer
                nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(D_HIDDEN * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # # 4th layer

                # nn.Conv2d(D_HIDDEN * 8, D_HIDDEN * 16, 4, 2, 1, bias=False),
                # nn.BatchNorm2d(D_HIDDEN * 16),
                # nn.LeakyReLU(0.2, inplace=True),
                # output layer
                nn.Conv2d(D_HIDDEN * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
#The activation functions used in the output layers of the generator and discriminator in a Generative Adversarial Network (GAN) are chosen based on the nature of the data and the roles of the networks.

# Generator (Tanh): The generator's goal is to generate data that resembles the real data. If the real data is images, these are often normalized to be in the range [-1, 1]. The Tanh activation function also outputs in the range [-1, 1], so it's a natural choice for the generator's output layer. This way, the output of the generator is already in the correct range to match the real data.

# Discriminator (Sigmoid): The discriminator's goal is to classify its input as real or fake. This is a binary classification problem, and the Sigmoid activation function is commonly used in the output layer for such problems. The Sigmoid function outputs a value in the range [0, 1], which can be interpreted as the probability of the input being real. A value close to 0 indicates a fake classification, and a value close to 1 indicates a real classification.

# These are common choices, but they're not the only possible ones. The best activation functions to use can depend on the specific characteristics of your data and model.

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def diversity_loss(fake):
    n = fake.size(0)
    fake = fake.view(n, -1)  # Flatten the images
    dists = torch.cdist(fake, fake, p=2)
    div_loss = -torch.mean(dists)  # Negative because we want to maximize diversity
    return div_loss

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def parse_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    output = {}
    for section in config.sections():
        output[section] = {}
        for key in config[section]:
            val_str = str(config[section][key])
            if(len(val_str)>0):
                val = parse_value_from_string(val_str) 
            else:
                val = None
            print(section, key,val_str, val)
            output[section][key] = val
    return output



def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    else:
        val = val_str
    return val

def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str=='True' or var_str == 'true' or var_str =='False' or var_str=='false'):
        return True
    else:
        return False

def parse_bool(var_str):
    if(var_str=='True' or var_str == 'true' ):
        return True
    else:
        return False
    
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False

def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        else:
            output.append(item)
    return output 
class MaskedCNN(nn.Conv2d):
	"""
	Implementation of Masked CNN Class as explained in A Oord et. al. 
	Taken from https://github.com/jzbontar/pixelcnn-pytorch
	"""

	def __init__(self, mask_type, *args, **kwargs):
		self.mask_type = mask_type
		assert mask_type in ['A', 'B'], "Unknown Mask Type"
		super(MaskedCNN, self).__init__(*args, **kwargs)
		self.register_buffer('mask', self.weight.data.clone())

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1)
		if mask_type =='A':
			self.mask[:,:,height//2,width//2:] = 0
			self.mask[:,:,height//2+1:,:] = 0
		else:
			self.mask[:,:,height//2,width//2+1:] = 0
			self.mask[:,:,height//2+1:,:] = 0


	def forward(self, x):
		self.weight.data*=self.mask
		return super(MaskedCNN, self).forward(x)
	
class PixelCNN(nn.Module):
	"""
	Network of PixelCNN as described in A Oord et. al. 
	"""
	def __init__(self, no_layers=8, kernel = 7, channels=64, device=None, lr = 1e-4):
		super(PixelCNN, self).__init__()
		self.no_layers = no_layers
		self.kernel = kernel
		self.channels = channels
		self.layers = {}
		self.device = device
		self.lr = lr

		self.Conv2d_1 = MaskedCNN('A',1,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
		self.ReLU_1= nn.ReLU(True)

		self.Conv2d_2 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
		self.ReLU_2= nn.ReLU(True)

		self.Conv2d_3 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
		self.ReLU_3= nn.ReLU(True)

		self.Conv2d_4 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
		self.ReLU_4= nn.ReLU(True)

		self.Conv2d_5 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
		self.ReLU_5= nn.ReLU(True)

		self.Conv2d_6 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_6 = nn.BatchNorm2d(channels)
		self.ReLU_6= nn.ReLU(True)

		self.Conv2d_7 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_7 = nn.BatchNorm2d(channels)
		self.ReLU_7= nn.ReLU(True)

		self.Conv2d_8 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
		self.BatchNorm2d_8 = nn.BatchNorm2d(channels)
		self.ReLU_8= nn.ReLU(True)
	
		if lr == 1e-5:
			self.Conv2d_9 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
			self.BatchNorm2d_9 = nn.BatchNorm2d(channels)
			self.ReLU_9= nn.ReLU(True)

			self.Conv2d_10 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
			self.BatchNorm2d_10 = nn.BatchNorm2d(channels)
			self.ReLU_10 = nn.ReLU(True)
		

		self.out = nn.Conv2d(channels, 256, 1)

	def forward(self, x):
		x = self.Conv2d_1(x)
		x = self.BatchNorm2d_1(x)
		x = self.ReLU_1(x)

		x = self.Conv2d_2(x)
		x = self.BatchNorm2d_2(x)
		x = self.ReLU_2(x)

		x = self.Conv2d_3(x)
		x = self.BatchNorm2d_3(x)
		x = self.ReLU_3(x)

		x = self.Conv2d_4(x)
		x = self.BatchNorm2d_4(x)
		x = self.ReLU_4(x)

		x = self.Conv2d_5(x)
		x = self.BatchNorm2d_5(x)
		x = self.ReLU_5(x)

		x = self.Conv2d_6(x)
		x = self.BatchNorm2d_6(x)
		x = self.ReLU_6(x)

		x = self.Conv2d_7(x)
		x = self.BatchNorm2d_7(x)
		x = self.ReLU_7(x)

		x = self.Conv2d_8(x)
		x = self.BatchNorm2d_8(x)
		x = self.ReLU_8(x)

		if self.lr == 1e-5:
			x = self.Conv2d_9(x)
			x = self.BatchNorm2d_9(x)
			x = self.ReLU_9(x)

			x = self.Conv2d_10(x)
			x = self.BatchNorm2d_10(x)
			x = self.ReLU_10(x)

		return self.out(x)
      
CUDA = True
DATA_PATH = './data'
BATCH_SIZE = 128
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 24
REAL_LABEL = 1
FAKE_LABEL = 0

lr = 1e-4
seed = 1

CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True

    # Example usage:
# os.chdir('..')
# os.chdir('..')
# root_dir = r'/data/s3287297/NIH_data/images'
# txt_file = r'/data/s3287297/NIH_data/train_val_list.txt'
# txt_file_test = r'/data/s3287297/NIH_data/test_list.txt'
root_dir = r'NIH_data/images'
txt_file = r'NIH_data/train_val_list.txt'
txt_file_test = r'NIH_data/test_list.txt'

# Define your transform
transform = transforms.Compose([
    transforms.Resize((X_DIM)),  # Resize image to (224, 224)
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor with mean and standard deviation
])

# Create custom dataset instance
custom_dataset = CustomImageDataset(folder=root_dir, txt_file=txt_file, trans=transform)
custom_dataset_test = CustomImageDataset(folder=root_dir, txt_file=txt_file_test, trans=transform)


# Define the size of your training and testing sets
train_size = custom_dataset # 80% of the dataset for training
test_size = custom_dataset_test  # 20% of the dataset for testing



# Create DataLoaders for training and testing sets
train_loader = torch.utils.data.DataLoader(train_size, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_size, batch_size=BATCH_SIZE, shuffle=True)


# Create the discriminator
img_size = 64
netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
            
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that I will use to visualize the progression of the generator
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0


# Create DataLoader
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=128, shuffle=True)
print("Starting Training Loop...")
print("Starting Testing Loop...")
# list_of_disc_errors_500 = [3.67794132232666, 3.503178119659424, 3.6076016426086426, 3.2208688259124756, 3.860383987426758, 3.8884263038635254, 3.5787596702575684, 4.286342144012451, 3.7707061767578125, 3.720978021621704, 3.254955530166626, 3.0980122089385986, 3.4185142517089844, 3.418964147567749, 3.989417314529419, 3.519407272338867, 3.5756218433380127, 4.06866455078125, 3.508878231048584, 3.6577486991882324, 3.6472702026367188, 4.0691304206848145, 3.7421486377716064, 3.6138362884521484, 4.099040508270264 ]
# list_of_disc_errors_300 = [69.31978607177734, 69.20040130615234, 69.24744415283203, 69.27139282226562, 69.21062469482422, 69.39337921142578, 69.55066680908203, 69.40667724609375, 69.22758483886719, 69.4830551147461, 69.5771713256836, 69.32005310058594, 69.31959533691406, 69.28053283691406, 69.39614868164062, 69.37401580810547, 69.03463745117188, 69.16797637939453, 69.31694793701172, 68.93363952636719, 69.4273910522461, 69.13243103027344, 69.41190338134766, 69.30957794189453, 69.27418518066406 ]
# list_of_disc_errors_baseline = [23.622438430786133, 24.197343826293945, 23.40094566345215, 23.98157501220703, 23.79994010925293, 23.728822708129883, 23.974998474121094, 23.890085220336914, 24.14176368713379, 23.74187660217285, 23.650381088256836, 24.00697135925293, 24.07979393005371, 23.579343795776367, 24.249757766723633, 23.928030014038086, 23.831193923950195, 23.691335678100586, 23.59296226501465, 23.849149703979492, 23.411224365234375, 23.728271484375,24.34131622314453, 23.822879791259766, 24.16730499267578 ]
list_of_epochs = np.arange(0, EPOCH_NUM, 1)
list_of_disc_errors_500 = np.zeros(EPOCH_NUM)
list_of_disc_errors_300 = np.zeros(EPOCH_NUM)
list_of_disc_errors_baseline = np.zeros(EPOCH_NUM)
fig, plt1 = plt.subplots()
plt1.set_xlabel('Epoch')
plt1.set_ylabel('Discriminator Loss')      
plt1.set_title("Discriminator Loss vs epoch Autoregressive model")
for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader, 0):
        img_size = 64
        netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
        netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))  
        real_cpu = data[0].to(DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=DEVICE)
        print(torch.Tensor.size(real_cpu))
        # if label.size(dim=0) != 64:
        #     top, left, bottom, right = 0, 0, 120, 120
        # else:
        #     top, left, bottom, right = 0, 0, 3, 2  # Example coordinates

        # Extract the region using slicing
        # This will extract the region from each image in the batch
        # real_cpu = real_cpu[top:bottom, left:right]
        # show_and_save_image(real_cpu[0], '/data/s3287297/NIH_VAE/after_slicing.png')
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)     
        # (2) Update the discriminator with fake data
        # Generate batch of latent vectors
        noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)

        with torch.no_grad():
            config_file = sys.argv[1]
            assert os.path.exists(config_file), "Configuration file does not exit!"
            config = parse_config(config_file)
            model = config['model']
            images = config['images']
            load_path = 'models_baseline_try_3/Model_Checkpoint_'+ str(epoch) +'.pt'
            if epoch == 24:
                load_path = 'models_baseline_try_3/Model_Checkpoint_Last.pt'
            # if epoch > 24 and epoch < 49:
            #     load_path = '/data/s3287297/last_try_autoreg/Model_Checkpoint_'+ str(epoch) +'_lr=0.0001img_dim=64.pt'
            # if epoch == 49:
            #     load_path = '/data/s3287297/last_try_autoreg/Model_Checkpoint_Last_lr=0.0001img_dim=64.pt'
            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
            no_images = images.get('no_images', 128)
            images_size = images.get('images_size', 64)
            images_channels = images.get('images_channels', 1)
            

            #Define and load model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = PixelCNN(channels = 64).to(device)
            if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            state_dict = torch.load(load_path)
            # Add 'module.' prefix to each key
            # new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
            net.load_state_dict(state_dict)
            net.eval()



            fake = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
            fake.fill_(0)

            #Generating images pixel by pixel
            for i in range(images_size):
                for j in range(images_size):
                    out = net(fake)
                    probs = F.softmax(out[:,:,i,j], dim=-1).data
                    fake[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

            #Saving images row wise
            torchvision.utils.save_image(fake, 'models_baseline_try_3/auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        print(output.size())
        print(label.size())
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        list_of_disc_errors_baseline[epoch] = errD.item()
        print(epoch)
        print(errD.item())
        # Open the file in append mode ('a') and write the loss value
        with open('models_baseline_try_3/loss_values.txt', 'a') as file:
            file.write(f"{errD.item()}\n")  # Add a newline character to separate entries


        with torch.no_grad():
            img_size = 64
            netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
            netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
            config_file = sys.argv[2]
            assert os.path.exists(config_file), "Configuration file does not exit!"
            config = parse_config(config_file)
            model = config['model']
            images = config['images']
            
            load_path = 'models_5e-4/Model_Checkpoint_'+ str(epoch) +'.pt'
            if EPOCH_NUM == 24:
                load_path = 'models_5e-4/Model_Checkpoint_Last.pt'
            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
            no_images = images.get('no_images', 128)
            images_size = 64
            images_channels = images.get('images_channels', 1)
            

            #Define and load model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = PixelCNN(channels = 64, lr=5e-4).to(device)
            if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net.load_state_dict(torch.load(load_path))
            net.eval()



            fake = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
            fake.fill_(0)

            #Generating images pixel by pixel
            for i in range(images_size):
                for j in range(images_size):
                    out = net(fake)
                    probs = F.softmax(out[:,:,i,j], dim=-1).data
                    fake[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

            #Saving images row wise
            torchvision.utils.save_image(fake, 'models_5e-4/auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        print(output.size())
        print(label.size())
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        list_of_disc_errors_300[epoch] = errD.item()
        print(epoch)
        print(errD.item())
        # Open the file in append mode ('a') and write the loss value
        with open('models_5e-4/loss_values.txt', 'a') as file:
            file.write(f"{errD.item()}\n")  # Add a newline character to separate entries



        with torch.no_grad():
            img_size = 64
            netD = Discriminator(IMAGE_CHANNEL, D_HIDDEN, img_size).to(DEVICE)
            netD.load_state_dict(torch.load('disc_64/discriminator_49.pth'))
            config_file = sys.argv[3]
            assert os.path.exists(config_file), "Configuration file does not exit!"
            config = parse_config(config_file)
            model = config['model']
            images = config['images']
            
            load_path = 'models_1e-5/Model_Checkpoint_'+ str(epoch) +'.pt'
            if EPOCH_NUM == 24:
                load_path = 'models_1e-5/Model_Checkpoint_Last.pt'
            assert os.path.exists(load_path), 'Saved Model File Does not exist!'
            no_images = images.get('no_images', 128)
            images_size = 64
            images_channels = images.get('images_channels', 1)
            

            #Define and load model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net = PixelCNN(channels = 64, lr=1e-5).to(device)
            if torch.cuda.device_count() > 1: #Accelerate testing if multiple GPUs available
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            net.load_state_dict(torch.load(load_path))
            net.eval()



            fake = torch.Tensor(no_images, images_channels, images_size, images_size).to(device)
            fake.fill_(0)

            #Generating images pixel by pixel
            for i in range(images_size):
                for j in range(images_size):
                    out = net(fake)
                    probs = F.softmax(out[:,:,i,j], dim=-1).data
                    fake[:,:,i,j] = torch.multinomial(probs, 1).float() / 255.0

            #Saving images row wise
            torchvision.utils.save_image(fake, 'models_1e-5/auto_reg_epoch_'+ str(epoch)+'.png', nrow=12, padding=0)
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        print(output.size())
        print(label.size())
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        list_of_disc_errors_500[epoch] = errD.item()
        print(epoch)
        print(errD.item())
        # Open the file in append mode ('a') and write the loss value
        with open('models_1e-5/loss_values.txt', 'a') as file:
            file.write(f"{errD.item()}\n")  # Add a newline character to separate entries
        break
plt1.plot(list_of_epochs,list_of_disc_errors_300,label="Discriminator Loss of the autoreg model with lr = 5e-4")
plt1.plot(list_of_epochs,list_of_disc_errors_500,label="Discriminator Loss of the autoreg model with lr = 1e-5")
plt1.plot(list_of_epochs,list_of_disc_errors_baseline,label="Discriminator Loss of baseline")
plt1.legend()
fig.savefig("/data/s3287297/Discriminator_vs_autoreg_lr_attempt_2.png",dpi=300)