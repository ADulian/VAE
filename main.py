import torch
import cv2

from vae import Data_Manager, VAE

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    device_id = '0'

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_id))

    # Data Manager
    data_manager = Data_Manager()

    # VAE
    model = VAE(data_manager, device, epochs=10)
    model.fit()

    # Single Sample
    model.load_model('vae_state.pt')
    sample = model.sample()

    cv2.imwrite('sample_test.png', sample)





