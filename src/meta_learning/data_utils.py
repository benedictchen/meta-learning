"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

Essential Data Utilities for Meta-Learning
=========================================

Core utilities for episode creation, data partitioning, and dataset management.
Implements essential functions missing from our package to compete with established libraries.
"""
import torch
import requests
import tqdm
from typing import Tuple, Optional, Union


class InfiniteIterator:
    """
    Infinitely loops over a given iterator.
    
    Simple but effective implementation matching learn2learn's approach.
    
    Example:
        >>> dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
        >>> inf_dataloader = InfiniteIterator(dataloader)
        >>> for iteration in range(10000):  # guaranteed to reach 10,000
        ...     X, y = next(inf_dataloader)
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)


class OnDeviceDataset(torch.utils.data.TensorDataset):
    """
    Converts an entire dataset into a TensorDataset, and optionally puts it on device.
    
    Useful to accelerate training with relatively small datasets.
    If the device is cpu and cuda is available, the TensorDataset will live in pinned memory.
    
    Args:
        dataset: Dataset to put on a device
        device: Device of dataset. Defaults to CPU
        transform: Transform to apply on the first variate of the dataset's samples X
        
    Example:
        >>> from torchvision import transforms
        >>> transforms = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     transforms.Normalize((0.1307,), (0.3081,)),
        ...     lambda x: x.view(1, 28, 28),
        ... ])
        >>> mnist = MNIST('~/data')
        >>> mnist_ondevice = OnDeviceDataset(mnist, device='cuda', transform=transforms)
    """

    def __init__(self, dataset, device=None, transform=None):
        data = []
        labels = []
        for x, y in dataset:
            data.append(x.unsqueeze(0))
            labels.append(y)
        
        data = torch.cat(data, dim=0)
        labels = torch.tensor(labels)
        
        if transform is not None:
            data = transform(data)
        
        if device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(device)
            
        if device.type == 'cpu' and torch.cuda.is_available():
            data = data.pin_memory()
            labels = labels.pin_memory()
        else:
            data = data.to(device)
            labels = labels.to(device)
            
        super(OnDeviceDataset, self).__init__(data, labels)


def download_file(source: str, destination: str, size: Optional[int] = None):
    """Download file with progress bar."""
    CHUNK_SIZE = 1 * 1024 * 1024
    
    if size is None:
        size = 0
    req = requests.get(source, stream=True)
    with open(destination, 'wb') as archive:
        for chunk in tqdm.tqdm(
            req.iter_content(chunk_size=CHUNK_SIZE),
            total=size // CHUNK_SIZE,
            leave=False,
        ):
            if chunk:
                archive.write(chunk)


def download_file_from_google_drive(id: str, destination: str):
    """Download file from Google Drive."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    """Get confirmation token from Google Drive response."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination: str):
    """Save response content to destination file."""
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)