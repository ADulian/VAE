import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets,transforms

# --------------------------------------------------------------------------------
class Data_Manager:
    def __init__(self, batch_size=64):

        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.train_set = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        self.test_set = datasets.MNIST('./data', train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

# --------------------------------------------------------------------------------
class VAE(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, data_mananger, device, hidden_dim=256, latent_dim=64, epochs=1, lr=1e-3):
        super(VAE, self).__init__()

        self.data_manager = data_mananger
        self.device = device

        self.epochs = epochs
        self.lr = lr

        self.input_dim = 28*28
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder-Decoder
        self.enc = Encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.dec = Decoder(self.input_dim, self.hidden_dim, self.latent_dim)

        # Optim and Criterion
        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = VAE_Criterion()

        # To device
        self.to(self.device)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        # Encode
        mu, log_var = self.enc(x)

        # Sample
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)

        z = eps.mul(std).add_(mu)

        # Decode
        rec = self.dec(z)

        return rec, mu, log_var

    # --------------------------------------------------------------------------------
    def fit(self):
        print("--- Training...")
        best_test_loss = float('inf')

        for i in range(self.epochs):
            train_loss = self.epoch_train()
            val_loss = self.epoch_val()

            train_loss /= len(self.data_manager.train_set)
            val_loss /= len(self.data_manager.test_set)

            print("Epoch {}/{}, Train Loss: {:.2f}, Test Loss: {:.2f}".format(i+1, self.epochs, train_loss, val_loss))

            if best_test_loss > val_loss:
                best_test_loss = val_loss
                torch.save(self.state_dict(), 'vae_state.pt')

    # --------------------------------------------------------------------------------
    def epoch_train(self):
        self.train()

        train_loss =0

        for i, data in enumerate(self.data_manager.train_loader):
            x, _ = data

            # Flatten image
            x = x.view(-1, self.input_dim)
            x = x.to(self.device)

            # Zero Grad
            self.optim.zero_grad()

            # Forward-Backward
            reconstruction, mu, log_var = self.forward(x)

            loss = self.criterion(x, reconstruction, mu, log_var)
            loss.backward()

            train_loss += loss.item()

            self.optim.step()

        return train_loss

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def epoch_val(self):
        self.eval()

        eval_loss = 0

        for i, data in enumerate(self.data_manager.test_loader):
            x, _ = data

            # Flatten image
            x = x.view(-1, self.input_dim)
            x = x.to(self.device)

            # Forward-Backward
            reconstruction, mu, log_var = self.forward(x)

            loss = self.criterion(x, reconstruction, mu, log_var)

            eval_loss += loss.item()

        return eval_loss

    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, inverse_norm=True):
        z = torch.randn(1, self.latent_dim).to(self.device)

        rec = self.dec(z)
        rec = rec.view(28,28).cpu().numpy()

        if inverse_norm:
            rec = rec * 255.0

        return rec

    # --------------------------------------------------------------------------------
    def load_model(self, model_name):
        self.load_state_dict(torch.load(model_name, map_location=self.device))

# --------------------------------------------------------------------------------
class Encoder(nn.Module):
    # --------------------------------------------------------------------------------
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.linear = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(True))

        # Q(z|x)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    # --------------------------------------------------------------------------------
    def forward(self, x):
        hidden = self.linear(x)

        mu = self.mu(hidden)
        log_var = self.log_var(hidden)

        return mu, log_var

# --------------------------------------------------------------------------------
class Decoder(nn.Module):
    # --------------------------------------------------------------------------------
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()

        # P(x|z)
        self.dec_latent = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                        nn.ReLU(True),
                                        nn.Linear(hidden_dim, output_dim),
                                        nn.Sigmoid())

    # --------------------------------------------------------------------------------
    def forward(self, x):
        reconstructed_sample = self.dec_latent(x)

        return reconstructed_sample

# --------------------------------------------------------------------------------
class VAE_Criterion(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self):
        super(VAE_Criterion, self).__init__()

        self.rec_criterion = nn.BCELoss(reduction='sum')  # Try MSE and compare results

    # --------------------------------------------------------------------------------
    def forward(self, x, rec, mu, log_var):
        rec_loss = self.rec_criterion(rec, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return rec_loss + kl_loss





