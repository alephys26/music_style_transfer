import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
from models import Generator, Discriminator

# Argument parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--name', type=str, default='experiment')
    return parser.parse_args()

args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
gen_A2B = Generator().to(device)
gen_B2A = Generator().to(device)
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)

# Define loss functions
criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers
g_optimizer = optim.Adam(list(gen_A2B.parameters()) + list(gen_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()), lr=args.lr, betas=(0.5, 0.999))

# Load Data
dataloader_A = torch.load("train_loader_beethoven.pth")
dataloader_B = torch.load("train_loader_chopin.pth")
dataloader = zip(dataloader_A, dataloader_B)

# Training Loop
def train():
    log_data = []
    for epoch in range(args.num_epochs):
        for i, ((real_A, label_A, mask_A), (real_B, label_B, mask_B)) in enumerate(tqdm(dataloader)):
            real_A, real_B = real_A.to(device), real_B.to(device)
            mask_A, mask_B = mask_A.to(device), mask_B.to(device)
            
            
            # Labels
            valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1), requires_grad=False).to(device)
            
            # Train Generators
            g_optimizer.zero_grad()
            
            fake_B = gen_A2B(real_A)
            fake_A = gen_B2A(real_B)
            
            loss_GAN_A2B = criterion_GAN(disc_B(fake_B), valid)
            loss_GAN_B2A = criterion_GAN(disc_A(fake_A), valid)
            
            recovered_A = gen_B2A(fake_B)
            recovered_B = gen_A2B(fake_A)
            
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)
            
            identity_A = gen_B2A(real_A)
            identity_B = gen_A2B(real_B)
            
            loss_identity_A = criterion_identity(identity_A, real_A)
            loss_identity_B = criterion_identity(identity_B, real_B)
            
            
            g_loss = (
                loss_GAN_A2B + loss_GAN_B2A + 
                args.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                args.lambda_identity * (loss_identity_A + loss_identity_B)
            )
            g_loss.backward()
            g_optimizer.step()
            
            # Train Discriminators
            d_optimizer.zero_grad()
            
            loss_real_A = criterion_GAN(disc_A(real_A), valid)
            loss_real_B = criterion_GAN(disc_B(real_B), valid)
            loss_fake_A = criterion_GAN(disc_A(fake_A.detach()), fake)
            loss_fake_B = criterion_GAN(disc_B(fake_B.detach()), fake)
            
            d_loss_A = (loss_real_A + loss_fake_A) / 2
            d_loss_B = (loss_real_B + loss_fake_B) / 2
            d_loss = d_loss_A + d_loss_B
            d_loss.backward()
            d_optimizer.step()
            
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | G Loss: {g_loss.item()} | D Loss: {d_loss.item()}")
        
        if args.log:
            log_data.append([epoch+1, g_loss.item(), d_loss.item(), (loss_cycle_A + loss_cycle_B).item(), (loss_identity_A + loss_identity_B).item()])
            df = pd.DataFrame(log_data, columns=["Epoch", "Generator Loss", "Discriminator Loss", "Cycle Loss", "Identity Loss"])
            df.to_csv(f"training_log_{args.name}.csv", index=False)
        
        if args.save_model:
            torch.save(gen_A2B.state_dict(), f"gen_A2B_{args.name}_epoch_{epoch+1}.pth")
            torch.save(gen_B2A.state_dict(), f"gen_B2A_{args.name}_epoch_{epoch+1}.pth")
            torch.save(disc_A.state_dict(), f"disc_A_{args.name}_epoch_{epoch+1}.pth")
            torch.save(disc_B.state_dict(), f"disc_B_{args.name}_epoch_{epoch+1}.pth")

train()
