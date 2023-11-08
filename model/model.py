from model.ae import AE
from model.vae import VAE
from model.ra import RA


def get_model(config):
    print(f"Loading model {config['model_name']}")
    if config['model_name'] == 'AE':
        return AE(config)
    elif config['model_name'] == 'VAE':
        return VAE(config)
    elif config['model_name'] == 'RA':
        return RA(config)
    else:
        raise ValueError(f"Unknown model name {config['model_name']}")
