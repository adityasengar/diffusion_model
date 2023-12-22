import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def forward_process(image, t, alphas_cumprod):
    """Applies noise to an image at a specific timestep t."""
    noise = torch.randn_like(image)
    
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])
    
    noisy_image = sqrt_alphas_cumprod_t * image + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

def main():
    """Demonstrates the forward noising process."""
    # Settings
    IMG_SIZE = 64
    TIMESTEPS = 300
    
    # Example Image (Create a dummy one for demonstration)
    try:
        # Try to open a real image if one exists
        image = Image.open("dummy_image.png").convert("RGB")
    except FileNotFoundError:
        print("dummy_image.png not found. Creating a placeholder image.")
        image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'red')

    # Transforms
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1) # Scale to [-1, 1]
    ])
    image_tensor = transform(image)

    # Prepare diffusion schedule
    betas = linear_beta_schedule(timesteps=TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # Show noisy images at different timesteps
    fig = plt.figure(figsize=(15, 3))
    for i, t in enumerate([0, 50, 100, 150, 299]):
        noisy_img = forward_process(image_tensor, torch.tensor([t]), alphas_cumprod)
        
        # Reverse transform for visualization
        reverse_transform = T.Compose([
            T.Lambda(lambda t: (t + 1) / 2),
            T.ToPILImage()
        ])
        
        ax = fig.add_subplot(1, 5, i + 1)
        ax.set_title(f"Timestep {t}")
        ax.imshow(reverse_transform(noisy_img))
        ax.axis('off')
        
    plt.savefig("forward_process_demo.png")
    print("Saved forward process demonstration to forward_process_demo.png")

if __name__ == "__main__":
    main()
