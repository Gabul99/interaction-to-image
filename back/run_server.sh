#!/bin/bash
# Run backend server on remote GPU server

cd /home/ella/courses/HCI/interaction-to-image

# Activate Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hci_i2i

# Run server using main.py (recommended)
echo "Starting backend server..."
python back/main.py
