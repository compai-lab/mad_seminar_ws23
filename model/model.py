from model.ae import AE
from model.vae import VAE


def get_model(config):
    print(f"Loading model {config['model_name']}")
    if config['model_name'] == 'AE':
        return AE(config)
    elif config['model_name'] == 'VAE':
        return VAE(config)
    else:
        raise ValueError(f"Unknown model name {config['model_name']}")
