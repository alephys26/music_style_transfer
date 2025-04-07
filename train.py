import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import argparse
from models import Generator, Discriminator

# --------------------------
# Argument Parser
# --------------------------
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

# --------------------------
# Model Hyperparameters
# --------------------------
VOCAB_SIZE = 282   # fixed vocab size
SEQ_LEN = 128    # fixed token sequence length

# --------------------------
# Instantiate Models
# --------------------------
gen_A2B = Generator(VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
gen_B2A = Generator(VOCAB_SIZE, seq_len=SEQ_LEN).to(device)
disc_A = Discriminator(VOCAB_SIZE).to(device)
disc_B = Discriminator(VOCAB_SIZE).to(device)

# --------------------------
# Loss Functions & Optimizers
# --------------------------
criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

g_optimizer = optim.Adam(
    list(gen_A2B.parameters()) + list(gen_B2A.parameters()),
    lr=args.lr, betas=(0.5, 0.999)
)
d_optimizer = optim.Adam(
    list(disc_A.parameters()) + list(disc_B.parameters()),
    lr=args.lr, betas=(0.5, 0.999)
)

# --------------------------
# Data Loading
# --------------------------
def get_beethoven(score,tok_seq,file_path):
  return torch.LongTensor([0])

def get_chopin(score,tok_seq,path):
  return torch.LongTensor([1])

dataloader_A = torch.load("train_loader_beethoven.pt", weights_only=False)
dataloader_B = torch.load("train_loader_chopin.pt", weights_only=False)
dataloader = zip(dataloader_A, dataloader_B)

# --------------------------
# Training Loop
# --------------------------
def train():
    log_data = []

    for epoch in range(args.num_epochs):
        for i, (batch_A, batch_B) in enumerate(tqdm(dataloader)):

            # Get real token sequences: [B, SEQ_LEN]
            real_A = batch_A['input_ids'].to(device).long()
            real_B = batch_B['input_ids'].to(device).long()

            valid = torch.ones((real_A.size(0), 1), device=device)
            fake = torch.zeros((real_A.size(0), 1), device=device)

            # ------------------
            # Train Generators
            # ------------------
            g_optimizer.zero_grad()

            # Generate fake tokens
            fake_B_logits = gen_A2B(real_A)   # returns logits since training mode is True
            fake_B = torch.argmax(fake_B_logits, dim=-1)  # convert to hard tokens

            fake_A_logits = gen_B2A(real_B)
            fake_A = torch.argmax(fake_A_logits, dim=-1)

            # GAN loss: Discriminators expect token inputs
            loss_GAN_A2B = criterion_GAN(disc_B(fake_B), valid)
            loss_GAN_B2A = criterion_GAN(disc_A(fake_A), valid)

            # Cycle Consistency:
            # Reconstruct real_A from fake_B and vice versa
            recovered_A_logits = gen_B2A(fake_B)
            recovered_A = torch.argmax(recovered_A_logits, dim=-1)

            recovered_B_logits = gen_A2B(fake_A)
            recovered_B = torch.argmax(recovered_B_logits, dim=-1)

            # For L1 loss, we convert token sequences to embeddings using the generator's embedding layer.
            # Using gen_A2B.embedding for A and gen_B2A.embedding for B.
            rec_A_embed = gen_A2B.embedding(recovered_A)   # [B, SEQ_LEN, D]
            real_A_embed = gen_A2B.embedding(real_A)

            rec_B_embed = gen_B2A.embedding(recovered_B)
            real_B_embed = gen_B2A.embedding(real_B)

            loss_cycle_A = criterion_cycle(rec_A_embed, real_A_embed)
            loss_cycle_B = criterion_cycle(rec_B_embed, real_B_embed)

            # Identity Loss:
            identity_A_logits = gen_B2A(real_A)
            identity_A = torch.argmax(identity_A_logits, dim=-1)
            identity_B_logits = gen_A2B(real_B)
            identity_B = torch.argmax(identity_B_logits, dim=-1)

            identity_A_embed = gen_A2B.embedding(identity_A)
            identity_B_embed = gen_B2A.embedding(identity_B)

            loss_identity_A = criterion_identity(identity_A_embed, real_A_embed)
            loss_identity_B = criterion_identity(identity_B_embed, real_B_embed)

            g_loss = (
                loss_GAN_A2B + loss_GAN_B2A +
                args.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
                args.lambda_identity * (loss_identity_A + loss_identity_B)
            )
            g_loss.backward()
            g_optimizer.step()

            # ------------------
            # Train Discriminators
            # ------------------
            d_optimizer.zero_grad()

            loss_real_A = criterion_GAN(disc_A(real_A), valid)
            loss_real_B = criterion_GAN(disc_B(real_B), valid)
            loss_fake_A = criterion_GAN(disc_A(fake_A.detach()), fake)
            loss_fake_B = criterion_GAN(disc_B(fake_B.detach()), fake)

            d_loss_A = 0.5 * (loss_real_A + loss_fake_A)
            d_loss_B = 0.5 * (loss_real_B + loss_fake_B)
            d_loss = d_loss_A + d_loss_B
            d_loss.backward()
            d_optimizer.step()

            del real_A, real_B, fake_A, fake_B, fake_A_logits, fake_B_logits 
            del recovered_A, recovered_B, recovered_A_logits, recovered_B_logits
            del identity_A, identity_B, identity_A_embed, identity_B_embed
            del identity_A_logits, identity_B_logits, rec_A_embed, rec_B_embed
            del real_A_embed, real_B_embed
            torch.cuda.empty_cache()

        print(f"[Epoch {epoch+1}/{args.num_epochs}] "
              f"G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

        # Logging
        if args.log:
            log_data.append([
                epoch + 1,
                g_loss.item(),
                d_loss.item(),
                (loss_cycle_A + loss_cycle_B).item(),
                (loss_identity_A + loss_identity_B).item()
            ])
            df = pd.DataFrame(
                log_data,
                columns=["Epoch", "Generator Loss", "Discriminator Loss", "Cycle Loss", "Identity Loss"]
            )
            df.to_csv(f"training_log_{args.name}.csv", index=False)

        # Save Models
        if args.save_model:
            torch.save(gen_A2B.state_dict(), f"gen_A2B_{args.name}_epoch_{epoch+1}.pth")
            torch.save(gen_B2A.state_dict(), f"gen_B2A_{args.name}_epoch_{epoch+1}.pth")
            torch.save(disc_A.state_dict(), f"disc_A_{args.name}_epoch_{epoch+1}.pth")
            torch.save(disc_B.state_dict(), f"disc_B_{args.name}_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()

