"""
Download datasets for AI Music Detector training.

Downloads:
- FMA Medium dataset (22 GB) - Real music
- SONICS dataset (~150 GB) - AI-generated music (Suno/Udio)

Usage:
    python download_data.py --dataset all
    python download_data.py --dataset fma
    python download_data.py --dataset sonics
    python download_data.py --dataset all --keep-zips  # Don't delete zip files
"""

import os
import argparse
import hashlib
import zipfile
import subprocess
import shutil
import requests
from pathlib import Path
from tqdm import tqdm
import yaml


def find_7zip():
    """Find 7-Zip executable if available."""
    # Common 7-Zip locations
    possible_paths = [
        # Windows
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        # Linux/macOS (if installed via package manager)
        "/usr/bin/7z",
        "/usr/local/bin/7z",
    ]
    
    # Check if 7z is in PATH
    if shutil.which("7z"):
        return "7z"
    
    # Check if 7zz is in PATH (newer versions on some systems)
    if shutil.which("7zz"):
        return "7zz"
    
    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


# Cache 7-Zip path
_7ZIP_PATH = find_7zip()
if _7ZIP_PATH:
    print(f"[INFO] Found 7-Zip: {_7ZIP_PATH} (will use for faster extraction)")
else:
    print("[INFO] 7-Zip not found, using Python zipfile (slower for large archives)")


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path, desc: str = None, expected_sha1: str = None):
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        if expected_sha1:
            print("Verifying checksum...")
            if verify_sha1(dest_path, expected_sha1):
                print("Checksum verified!")
                return True
            else:
                print("Checksum mismatch, re-downloading...")
                dest_path.unlink()
        else:
            return True
    
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    if expected_sha1:
        print("Verifying checksum...")
        if not verify_sha1(dest_path, expected_sha1):
            raise ValueError(f"Checksum verification failed for {dest_path}")
        print("Checksum verified!")
    
    return True


def verify_sha1(file_path: Path, expected_sha1: str) -> bool:
    """Verify SHA1 checksum of a file."""
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest() == expected_sha1


def extract_zip(zip_path: Path, dest_dir: Path, delete_after: bool = True):
    """Extract a zip file and optionally delete it after.
    
    Uses 7-Zip if available (faster and handles edge cases better),
    otherwise falls back to Python's zipfile module.
    """
    print(f"Extracting {zip_path.name}...")
    
    success = False
    
    if _7ZIP_PATH:
        # Use 7-Zip for extraction (faster, especially for large files)
        try:
            print(f"  Using 7-Zip: {_7ZIP_PATH}")
            # 7z x archive.zip -oOutputDir -y
            # x = extract with full paths
            # -o = output directory
            # -y = assume yes to all prompts
            cmd = [_7ZIP_PATH, "x", str(zip_path), f"-o{dest_dir}", "-y"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                success = True
                print(f"  Extracted to {dest_dir}")
            else:
                print(f"  7-Zip failed: {result.stderr}")
                print("  Falling back to Python zipfile...")
        except Exception as e:
            print(f"  7-Zip error: {e}")
            print("  Falling back to Python zipfile...")
    
    if not success:
        # Fallback to Python's zipfile
        print("  Using Python zipfile...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(dest_dir)
            success = True
            print(f"  Extracted to {dest_dir}")
        except zipfile.BadZipFile as e:
            print(f"  ERROR: Bad zip file - {e}")
            print("  Try installing 7-Zip for better compatibility:")
            print("    Windows: https://www.7-zip.org/")
            print("    Linux: sudo apt install p7zip-full")
            print("    macOS: brew install p7zip")
            raise
    
    if delete_after and success:
        print(f"  Deleting {zip_path.name} to save disk space...")
        zip_path.unlink()
        print(f"  Deleted {zip_path.name}")


def is_fma_extracted(fma_dir: Path) -> bool:
    """Check if FMA dataset is already extracted."""
    fma_medium = fma_dir / "fma_medium"
    fma_metadata = fma_dir / "fma_metadata"
    
    # Check if directories exist and have content
    if not fma_medium.exists() or not fma_metadata.exists():
        return False
    
    # Check for some expected files/folders
    # FMA medium has numbered folders like 000, 001, etc.
    subdirs = list(fma_medium.iterdir()) if fma_medium.exists() else []
    if len(subdirs) < 10:  # Should have many numbered folders
        return False
    
    # Check metadata has expected files
    tracks_csv = fma_metadata / "tracks.csv"
    if not tracks_csv.exists():
        return False
    
    return True


def is_sonics_extracted(sonics_dir: Path) -> bool:
    """Check if SONICS dataset is already downloaded."""
    if not sonics_dir.exists():
        return False
    
    # Check for fake_songs directory or csv files
    fake_songs = sonics_dir / "fake_songs"
    fake_songs_csv = sonics_dir / "fake_songs.csv"
    
    # Either directory with songs or the metadata should exist
    if fake_songs.exists():
        # Check if it has some mp3 files
        mp3_files = list(fake_songs.glob("*.mp3"))
        if len(mp3_files) > 100:  # Should have many files
            return True
    
    # If we have the metadata, HuggingFace download might be in progress or complete
    if fake_songs_csv.exists():
        return True
    
    return False


def download_fma(data_dir: Path, keep_zips: bool = False):
    """Download FMA Medium dataset."""
    print("\n" + "="*60)
    print("Downloading FMA Medium Dataset")
    print("="*60)
    
    fma_dir = data_dir / "fma"
    fma_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    if is_fma_extracted(fma_dir):
        print("\nFMA dataset already downloaded and extracted!")
        print(f"  Real music tracks: {fma_dir / 'fma_medium'}")
        print(f"  Metadata: {fma_dir / 'fma_metadata'}")
        print("Skipping download. Use --force to re-download.")
        return
    
    # FMA URLs and checksums
    files = {
        "fma_metadata.zip": {
            "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
            "sha1": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
            "size": "342 MB",
            "extract_dir": "fma_metadata"
        },
        "fma_medium.zip": {
            "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
            "sha1": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
            "size": "22 GB",
            "extract_dir": "fma_medium"
        }
    }
    
    for filename, info in files.items():
        zip_path = fma_dir / filename
        extract_dir = fma_dir / info["extract_dir"]
        
        # Skip if already extracted
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"\n{info['extract_dir']} already extracted, skipping...")
            # Clean up zip if it exists
            if zip_path.exists() and not keep_zips:
                print(f"Cleaning up {filename}...")
                zip_path.unlink()
            continue
        
        # Download
        download_file(
            info["url"], 
            zip_path, 
            desc=f"{filename} ({info['size']})",
            expected_sha1=info["sha1"]
        )
        
        # Extract and delete zip
        extract_zip(zip_path, fma_dir, delete_after=not keep_zips)
    
    print("\nFMA download complete!")
    print(f"Real music tracks: {fma_dir / 'fma_medium'}")
    print(f"Metadata: {fma_dir / 'fma_metadata'}")


def download_sonics(data_dir: Path):
    """Download SONICS dataset from HuggingFace."""
    print("\n" + "="*60)
    print("Downloading SONICS Dataset")
    print("="*60)
    
    sonics_dir = data_dir / "sonics"
    
    # Check if already downloaded
    if is_sonics_extracted(sonics_dir):
        print("\nSONICS dataset already downloaded!")
        print(f"  AI-generated music: {sonics_dir}")
        print("Skipping download. Delete the folder to re-download.")
        return
    
    sonics_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    print("Downloading from HuggingFace (this may take a while)...")
    print("Repository: awsaf49/sonics")
    
    snapshot_download(
        repo_id="awsaf49/sonics",
        repo_type="dataset",
        local_dir=str(sonics_dir),
        resume_download=True
    )
    
    print("\nSONICS download complete!")
    print(f"AI-generated music: {sonics_dir}")
    
    # Note about real songs
    print("\n" + "-"*60)
    print("NOTE: SONICS only contains fake songs directly.")
    print("For real songs, we use the FMA dataset instead.")
    print("-"*60)


def download_sonics_kaggle(data_dir: Path, keep_zips: bool = False):
    """Alternative: Download SONICS from Kaggle."""
    print("\n" + "="*60)
    print("Downloading SONICS Dataset from Kaggle")
    print("="*60)
    
    sonics_dir = data_dir / "sonics"
    
    # Check if already downloaded
    if is_sonics_extracted(sonics_dir):
        print("\nSONICS dataset already downloaded!")
        print(f"  AI-generated music: {sonics_dir}")
        print("Skipping download. Delete the folder to re-download.")
        return
    
    sonics_dir.mkdir(parents=True, exist_ok=True)
    
    print("Make sure you have kaggle API configured.")
    print("See: https://www.kaggle.com/docs/api")
    
    # Kaggle downloads and unzips automatically with --unzip
    result = os.system(f"kaggle datasets download -d awsaf49/sonics-dataset --unzip -p {sonics_dir}")
    
    if result == 0:
        # Clean up any leftover zip files
        if not keep_zips:
            for zip_file in sonics_dir.glob("*.zip"):
                print(f"Cleaning up {zip_file.name}...")
                zip_file.unlink()
    
    print(f"\nSONICS download complete: {sonics_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for AI Music Detector")
    parser.add_argument(
        "--dataset", 
        choices=["all", "fma", "sonics", "sonics-kaggle"],
        default="all",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip files after extraction (default: delete to save space)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    data_dir = Path(args.data_dir) if args.data_dir else Path(config["paths"]["data_dir"])
    data_dir = data_dir.resolve()
    
    print(f"Data directory: {data_dir}")
    
    if args.force:
        print("\n[!] Force mode: will re-download existing datasets")
    
    if args.dataset in ["all", "fma"]:
        if args.force:
            # Remove existing to force re-download
            fma_dir = data_dir / "fma"
            if fma_dir.exists():
                print(f"Removing existing FMA directory...")
                shutil.rmtree(fma_dir)
        download_fma(data_dir, keep_zips=args.keep_zips)
    
    if args.dataset in ["all", "sonics"]:
        if args.force:
            sonics_dir = data_dir / "sonics"
            if sonics_dir.exists():
                print(f"Removing existing SONICS directory...")
                shutil.rmtree(sonics_dir)
        download_sonics(data_dir)
    
    if args.dataset == "sonics-kaggle":
        if args.force:
            sonics_dir = data_dir / "sonics"
            if sonics_dir.exists():
                print(f"Removing existing SONICS directory...")
                shutil.rmtree(sonics_dir)
        download_sonics_kaggle(data_dir, keep_zips=args.keep_zips)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nNext step: Run extract_fakeprints.py to extract features")
    print(f"\nExample commands:")
    print(f"  python extract_fakeprints.py --input {data_dir}/fma/fma_medium --output ./output/fma_fakeprints.npy --label real")
    print(f"  python extract_fakeprints.py --input {data_dir}/sonics/fake_songs --output ./output/sonics_fakeprints.npy --label fake")


if __name__ == "__main__":
    main()
