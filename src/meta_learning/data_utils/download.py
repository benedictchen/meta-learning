"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Robust Dataset Download Utilities
=================================

Advanced download utilities with progress tracking, resume capability,
integrity verification, and error recovery.
"""

# Dataset Download System Implementation Complete

from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List, Union
import os
import time
import hashlib
import requests
from pathlib import Path
import warnings
from urllib.parse import urlparse
import threading
from dataclasses import dataclass
from tqdm import tqdm


CHUNK_SIZE = 1 * 1024 * 1024  # 1MB chunks


@dataclass
class DownloadProgress:
    """Progress tracking for downloads."""
    downloaded: int = 0
    total: int = 0
    speed: float = 0.0
    eta: Optional[int] = None
    
    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.downloaded / self.total


def download_file(
    source: str,
    destination: str,
    size: Optional[int] = None,
    checksum: Optional[str] = None,
    checksum_type: str = "md5",
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    resume: bool = True,
    max_retries: int = 3,
    timeout: int = 30
) -> bool:
    """
    Download file with progress tracking and resume capability.
    
    Args:
        source: URL to download from
        destination: Local path to save file
        size: Expected file size (for progress tracking)
        checksum: Expected checksum for verification
        checksum_type: Type of checksum (md5, sha256, etc.)
        progress_callback: Optional callback for progress updates
        resume: Whether to resume partial downloads
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        True if download successful, False otherwise
    """
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists and is complete
    if destination_path.exists():
        if checksum and verify_checksum(str(destination_path), checksum, checksum_type):
            return True
        elif not resume:
            destination_path.unlink()
    
    # Determine resume position
    resume_byte_pos = 0
    if resume and destination_path.exists():
        resume_byte_pos = destination_path.stat().st_size
    
    headers = {}
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    for attempt in range(max_retries):
        try:
            with requests.get(source, headers=headers, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                
                # Handle range request responses
                if resume_byte_pos > 0 and response.status_code not in (206, 200):
                    # Server doesn't support range requests
                    resume_byte_pos = 0
                    headers = {}
                    response = requests.get(source, headers=headers, stream=True, timeout=timeout)
                    response.raise_for_status()
                
                total_size = None
                if 'content-length' in response.headers:
                    content_length = int(response.headers['content-length'])
                    if resume_byte_pos > 0 and response.status_code == 206:
                        total_size = resume_byte_pos + content_length
                    else:
                        total_size = content_length
                elif size:
                    total_size = size
                
                # Initialize progress tracking
                progress = DownloadProgress(downloaded=resume_byte_pos, total=total_size or 0)
                
                # Open file for writing
                mode = 'ab' if resume_byte_pos > 0 else 'wb'
                with open(destination_path, mode) as f:
                    # Setup progress bar
                    with tqdm(
                        total=total_size,
                        initial=resume_byte_pos,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {destination_path.name}"
                    ) as pbar:
                        
                        start_time = time.time()
                        last_update = start_time
                        
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                chunk_size = len(chunk)
                                progress.downloaded += chunk_size
                                pbar.update(chunk_size)
                                
                                # Update speed and ETA
                                current_time = time.time()
                                if current_time - last_update >= 1.0:  # Update every second
                                    elapsed = current_time - start_time
                                    if elapsed > 0:
                                        progress.speed = (progress.downloaded - resume_byte_pos) / elapsed
                                        if progress.speed > 0 and total_size:
                                            remaining = total_size - progress.downloaded
                                            progress.eta = int(remaining / progress.speed)
                                    
                                    if progress_callback:
                                        progress_callback(progress)
                                    
                                    last_update = current_time
                
                # Verify download
                if checksum:
                    if not verify_checksum(str(destination_path), checksum, checksum_type):
                        warnings.warn(f"Checksum verification failed for {destination}")
                        if destination_path.exists():
                            destination_path.unlink()
                        continue
                
                return True
                
        except (requests.RequestException, IOError) as e:
            if attempt == max_retries - 1:
                warnings.warn(f"Download failed after {max_retries} attempts: {e}")
                if destination_path.exists():
                    destination_path.unlink()
                return False
            
            # Wait before retry with exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    return False


def verify_checksum(
    file_path: str,
    expected_checksum: str,
    checksum_type: str = "md5"
) -> bool:
    """
    Verify file integrity using checksum.
    
    Args:
        file_path: Path to file to verify
        expected_checksum: Expected checksum value
        checksum_type: Type of checksum (md5, sha1, sha256, sha512)
        
    Returns:
        True if checksum matches, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Select hash algorithm
    checksum_type = checksum_type.lower()
    if checksum_type == 'md5':
        hasher = hashlib.md5()
    elif checksum_type == 'sha1':
        hasher = hashlib.sha1()
    elif checksum_type == 'sha256':
        hasher = hashlib.sha256()
    elif checksum_type == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported checksum type: {checksum_type}")
    
    # Compute checksum
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
        
        actual_checksum = hasher.hexdigest().lower()
        expected_checksum = expected_checksum.lower()
        
        return actual_checksum == expected_checksum
        
    except IOError:
        return False


def download_from_google_drive(
    file_id: str,
    destination: str,
    checksum: Optional[str] = None,
    checksum_type: str = "md5",
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None
) -> bool:
    """
    Download file from Google Drive with token handling.
    
    Args:
        file_id: Google Drive file ID
        destination: Local path to save file
        checksum: Expected checksum for verification
        checksum_type: Type of checksum
        progress_callback: Optional callback for progress updates
        
    Returns:
        True if download successful, False otherwise
    """
    # Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    session = requests.Session()
    
    # First request to get confirmation token if needed
    response = session.get(url, params={'confirm': 't'}, stream=True)
    
    # Check for virus scan warning and extract confirmation token
    if 'confirm=' in response.url or 'virus scan warning' in response.text.lower():
        # Look for download confirmation token
        for line in response.text.split('\n'):
            if 'confirm=' in line and 'download' in line.lower():
                # Extract token from the line
                import re
                token_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', line)
                if token_match:
                    token = token_match.group(1)
                    # Use the token to get the actual download
                    download_url = f"https://drive.google.com/uc?confirm={token}&id={file_id}"
                    return download_file(
                        download_url, 
                        destination, 
                        checksum=checksum,
                        checksum_type=checksum_type,
                        progress_callback=progress_callback
                    )
                break
    
    # If no token needed, proceed with direct download
    return download_file(
        url,
        destination,
        checksum=checksum,
        checksum_type=checksum_type,
        progress_callback=progress_callback
    )


class AdvancedProgressBar:
    """
    Advanced progress bar with pause/resume and nested progress support.
    """
    
    def __init__(self, total: Optional[int] = None, desc: str = "Progress", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.paused = False
        self.start_time = time.time()
        self.last_update = self.start_time
        self._pbar = None
        
    def __enter__(self):
        self._pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        return self
        
    def __exit__(self, *args):
        if self._pbar:
            self._pbar.close()
    
    def update(self, n: int = 1):
        """Update progress by n units."""
        if not self.paused and self._pbar:
            self._pbar.update(n)
            self.current += n
    
    def pause(self):
        """Pause the progress bar."""
        self.paused = True
        if self._pbar:
            self._pbar.set_description(f"{self.desc} (PAUSED)")
    
    def resume(self):
        """Resume the progress bar."""
        self.paused = False
        if self._pbar:
            self._pbar.set_description(self.desc)
    
    def set_description(self, desc: str):
        """Update the progress bar description."""
        self.desc = desc
        if self._pbar and not self.paused:
            self._pbar.set_description(desc)


def batch_download(
    downloads: List[Dict[str, Any]],
    max_concurrent: int = 3,
    progress_callback: Optional[Callable] = None
) -> Dict[str, bool]:
    """
    Download multiple files concurrently.
    
    Args:
        downloads: List of download specifications, each containing:
                  - url: Source URL
                  - destination: Local path
                  - checksum: Optional checksum
                  - checksum_type: Optional checksum type
        max_concurrent: Maximum number of concurrent downloads
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary mapping destination paths to success status
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    
    def download_single(download_spec):
        url = download_spec['url']
        destination = download_spec['destination']
        checksum = download_spec.get('checksum')
        checksum_type = download_spec.get('checksum_type', 'md5')
        
        success = download_file(
            url,
            destination,
            checksum=checksum,
            checksum_type=checksum_type,
            progress_callback=progress_callback
        )
        
        return destination, success
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_download = {
            executor.submit(download_single, download): download['destination']
            for download in downloads
        }
        
        for future in as_completed(future_to_download):
            destination, success = future.result()
            results[destination] = success
    
    return results