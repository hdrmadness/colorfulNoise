import numpy as np
from PIL import Image
import random
import torch
from diffusers import AutoencoderKL
from torchvision import transforms as tfms

# create an image 1 of 3 ways and then send to vae (or not):  

#   1.) upload image
#   2.) create an RGB noised image (color is user selectable)
#   3.) create a multi-resolution noise image 

# opts: change these to see different image types

# generate an RGB noise image or multiresolution image if true, upload if false
makeSumNoise = True

# generate a multiresolution noise image
mrn = False

# send the image to the vae to make latent, also saves latent to disk
send2vae = False

def input_image_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return  latent.latent_dist.sample()*0.18215 

def generate_image(base_color, noise_color, dimensions, noise_mean, noise_variance, save_path):
    
    # Create an array with the base color
    base_array = np.array(base_color, dtype=np.uint8)
    
    # Create an image array with the base color
    image_array = np.ones((dimensions[1], dimensions[0], 3), dtype=np.uint8) * base_array
    
    # Generate noise with specific mean and variance
    noise = np.random.normal(noise_mean, np.sqrt(noise_variance), (dimensions[1], dimensions[0], 3))
    
    # Apply the colored noise to the image
    for i in range(3):
        image_array[:,:,i] += (noise[:,:,i] * noise_color[i]).astype(np.uint8)
    
    # Create an image from the array
    image = Image.fromarray(image_array, 'RGB')
    
    # Show the image
    image.show()
    
    # Save the image
    image.save(save_path)
    
    return image_array

def makeRn(x, discount=0.5):
    
    b, c, w, h = x.shape 
    u = torch.nn.Upsample(size=(w, h), mode='bilinear')
    noise = torch.randn_like(x)
    
    for i in range(10):
        r = random.random()*2+2 # Rather than always going 2x, 
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    
    return noise/noise.std()  #Scaled back to roughly unit variance


if makeSumNoise == True: 
    
    if mrn == True:
        
        #latent_tensor = torch.load('latent_tensor.pt')
        
        height = 64
        width = 64
        channels = 3  

        # Create an empty image array
        empty_image = np.empty((height, width, channels), dtype=np.uint8)

        # Convert to tensor and unsqueeze
        xr = tfms.ToTensor()(empty_image).unsqueeze(0)

        # call noise func
        xp = makeRn(xr)

        # Convert tensor back to PIL Image
        im = tfms.ToPILImage()(xp.squeeze(0).cpu())

        # Show the image
        im.show()

        # Save the image
        save_path = 'multiresolution-noise.png'
        im.save(save_path)
        
        print(" Multiresolution noise generated")         

        input_image = im

    else:   
        
        # Example usage:
        base_color = [90, 23, 133]  #  (RGB) base color violet
        noise_color = [255, 0, 0]  # Add Pure red noise (RGB)
        dimensions = (512, 512)  # Image dimensions (width, height)
        noise_mean = 0.0  # Mean of the noise
        noise_variance = 1  # Variance of the noise
        save_path = 'rgb-noise.png'  # Path to save the image

        rgbNoise = generate_image(base_color, noise_color, dimensions, noise_mean, noise_variance, save_path)
        
        print(" RGB noise generated")        
        
        input_image = rgbNoise
else: 
    
    # Load the input image
    path = r'C:\Users\hdrma\Documents\Python\imageAugmentation\cng.png'
    input_image = Image.open(path).resize((512, 512), resample=Image.Resampling.LANCZOS).convert('RGB')
    print(" User selected image loaded")

if send2vae == True:
    
    torch.manual_seed(0)
    torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    # To the GPU 
    vae = vae.to(torch_device)

    # returns torch.Size([1, 4, 64, 64])
    latent_tensor = input_image_to_latent(input_image) 

    # Move the latent tensor to the CPU device
    latent_tensor_cpu = latent_tensor.cpu()

    # Save the tensor directly as a file
    torch.save(latent_tensor_cpu, 'latent_tensor.pt')




