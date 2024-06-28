import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import os
from unet import Unet

# ------------------------------------------------------------------
# Diffusion Components
# ------------------------------------------------------------------

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_noisy_image(x_start, t, alphas_cumprod):
    """Returns a noisy image at a specific timestep t."""
    noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image, noise

# ------------------------------------------------------------------
# Sampling / Generation
# ------------------------------------------------------------------

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1, timesteps=300):
    """Generates a batch of images by reversing the diffusion process."""
    print("Sampling new images...")
    
    # Prepare diffusion schedule constants
    betas = linear_beta_schedule(timesteps=timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    device = next(model.parameters()).device
    
    # Start with random noise
    img = torch.randn(batch_size, channels, image_size, image_size, device=device)
    
    for i in reversed(range(0, timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        predicted_noise = model(img, t)
        
        # Denoise the image
        beta_t = betas[i].to(device)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[i]).to(device)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / alphas[i]).to(device)
        
        model_mean = sqrt_recip_alpha_t * (img - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if i != 0:
            posterior_variance_t = posterior_variance[i].to(device)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            img = model_mean
            
    return img

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train(model, data_loader, optimizer, epochs, device, timesteps, alphas_cumprod):
    """The main training loop."""
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        for step, (images, _) in enumerate(data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            noisy_images, noise = get_noisy_image(images, t, alphas_cumprod.to(device))
            predicted_noise = model(noisy_images, t)
            
            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f}")

# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
def main():
    """Main function to run the diffusion model training and sampling."""
    # Settings
    IMG_SIZE = 64
    BATCH_SIZE = 128
    TIMESTEPS = 300
    EPOCHS = 20
    OUTPUT_DIR = "output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset
    transforms_pipe = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    dataset = datasets.FashionMNIST(".", download=True, transform=transforms_pipe)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Diffusion Schedule
    alphas_cumprod = torch.cumprod(1. - linear_beta_schedule(timesteps=TIMESTEPS), axis=0)
    
    # Model and Optimizer
    model = Unet(dim=IMG_SIZE, channels=1, dim_mults=(1, 2, 4,))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    print("--- Starting Training ---")
    train(model, dataloader, optimizer, EPOCHS, device, TIMESTEPS, alphas_cumprod)
    print("--- Training Finished ---")

    # Save the model
    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Model saved to diffusion_model.pth")

    # Generate and save samples
    generated_images = sample(model, image_size=IMG_SIZE, batch_size=16, channels=1)
    
    # Reverse transform for visualization
    reverse_transform = transforms.Lambda(lambda t: (t + 1) / 2)
    grid = utils.make_grid(reverse_transform(generated_images), nrow=4)
    utils.save_image(grid, os.path.join(OUTPUT_DIR, "generated_samples.png"))
    print(f"Saved generated samples to {OUTPUT_DIR}/generated_samples.png")

if __name__ == "__main__":
    main()