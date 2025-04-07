import torch
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import miditok


def inference(test_loader, model, classifier):
    model.eval()

    with torch.no_grad():
        acc = 0
        c = 0
        for batch in test_loader:
            output_tokens = model(batch['input_ids'])  # shape: [B, T]
            output = classifier(output_tokens)
            for k in output:
                if k>=0.5:
                    acc += 1
                c+=1
        acc /= c
    print(f"Inference completed. Accuracy: {acc}")


if __name__ == "__main__":
    test_loader = torch.load("test_loader_chopin.pt", weights_only=False)

    gen_A2B = torch.load("gen_A2B_last_epoch_50.pt", weights_only=False)
    clf = torch.load("classifier.pt", weights_only=False)
    inference(test_loader, gen_A2B, clf)


