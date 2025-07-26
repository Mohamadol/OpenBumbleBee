#!/bin/bash

DISK="/dev/nvme0n1"  # Your SSD disk name
MOUNT_POINT="/mnt/home"
FILESYSTEM="ext4"

if [ ! -b "$DISK" ]; then
    echo "Disk $DISK does not exist. Please check the disk name."
    exit 1
fi

echo "WARNING: This will format $DISK and erase all data on it!"
read -p "Are you sure you want to proceed? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborting."
    exit 0
fi

if mountpoint -q "$DISK"; then
    echo "Unmounting $DISK"
    sudo umount "$DISK"
fi

echo "Formatting $DISK as $FILESYSTEM"
sudo mkfs."$FILESYSTEM" "$DISK"

if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point at $MOUNT_POINT"
    sudo mkdir -p "$MOUNT_POINT"
fi

echo "Mounting $DISK to $MOUNT_POINT"
sudo mount "$DISK" "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
    echo "Disk successfully mounted at $MOUNT_POINT"
else
    echo "Failed to mount the disk. Check the device and try again."
    exit 1
fi

UUID=$(sudo blkid -s UUID -o value "$DISK")
if [ -z "$UUID" ]; then
    echo "Unable to retrieve UUID for $DISK. Ensure the disk is formatted and try again."
    exit 1
fi

echo "Adding $DISK to /etc/fstab for automatic mounting"
echo "UUID=$UUID $MOUNT_POINT $FILESYSTEM defaults 0 0" | sudo tee -a /etc/fstab
echo "Disk was mounted!"

# Create directory on /mnt/home with sudo, then set ownership to the current user
echo "Creating and setting up directory at /mnt/home"
sudo mkdir -p /mnt/home
sudo chown -R "$(whoami)":"$(whoami)" /mnt/home
cd /mnt/home || exit

# Add PPAs
echo "Adding necessary PPAs..."
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y

# Update and install dependencies
echo "Updating and installing system dependencies..."
sudo apt update
sudo apt install -y build-essential cmake libgmp-dev libglib2.0-dev libssl-dev \
                    libboost-all-dev m4 opam unzip bubblewrap \
                    graphviz tmux bc time openssl


sudo apt install -y gcc-11 g++-11 libasan6 \
  git wget curl unzip autoconf make lld-15 \
  cmake ninja-build vim-common libgl1 libglib2.0-0
sudo apt clean
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100


CONDA_ARCH=x86_64
INSTALLER=Miniconda3-py310_24.3.0-0-Linux-$CONDA_ARCH.sh
PREFIX="$HOME/miniconda3"

# Download and install Miniconda to $HOME
wget https://repo.anaconda.com/miniconda/$INSTALLER \
  && bash $INSTALLER -b -p $PREFIX \
  && rm -f $INSTALLER \
  && $PREFIX/bin/conda init

# Add conda to path for current session
export PATH="$PREFIX/bin:$PATH"


BAZEL_ARCH=amd64 
mkdir -p "$HOME/bin"

# Download and install Bazelisk (as bazel) to user-local bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-$BAZEL_ARCH -O "$HOME/bin/bazel" \
  && chmod +x "$HOME/bin/bazel"

# Add to PATH and persist it in .bashrc
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/bin:$PATH"
