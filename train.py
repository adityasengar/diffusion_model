import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import Unet # Assuming unet.py is in the same directory

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
    # Settings
    IMG_SIZE = 64
    BATCH_SIZE = 128
    TIMESTEPS = 300
    EPOCHS = 20 # Keep it low for a demonstration
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    transforms_ = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale to [-1, 1]
    ])
    
    try:
        dataset = datasets.FashionMNIST(".", download=True, transform=transforms_)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"Could not download dataset. Please check your internet connection. Error: {e}")
        return

    # Diffusion Schedule
    betas = linear_beta_schedule(timesteps=TIMESTEPS)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # Model and Optimizer
    model = Unet(dim=IMG_SIZE, channels=1, dim_mults=(1, 2, 4,))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    print("--- Starting Training ---")
    train(model, dataloader, optimizer, EPOCHS, device, TIMESTEPS, alphas_cumprod)
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
