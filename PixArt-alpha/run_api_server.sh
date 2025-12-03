#!/bin/bash
# Run the PixArt-Alpha ControlNet REST API Server
#
# Usage:
#   ./run_api_server.sh [--port PORT] [--image_size SIZE] [--model_path PATH]
#
# Examples:
#   ./run_api_server.sh
#   ./run_api_server.sh --port 8080 --image_size 512
#   ./run_api_server.sh --model_path /path/to/custom/model.pth

# Default values
PORT=8000
IMAGE_SIZE=512
MODEL_PATH="/home/jaesang/interaction-to-image/PixArt-alpha/model/PixArt-XL-2-512-ControlNet.pth"
CONFIG_512="configs/pixart_app_config/PixArt_xl2_img512_controlHed.py"
CONFIG_1024="configs/pixart_app_config/PixArt_xl2_img1024_controlHed.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Select config based on image size
if [ "$IMAGE_SIZE" -eq 512 ]; then
    CONFIG="$CONFIG_512"
    DEFAULT_MODEL="PixArt-XL-2-512-ControlNet.pth"
else
    CONFIG="$CONFIG_1024"
    DEFAULT_MODEL="PixArt-XL-2-1024-ControlNet.pth"
fi

# Set model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/home/jaesang/interaction-to-image/PixArt-alpha/model/$DEFAULT_MODEL"
fi

echo "=========================================="
echo "PixArt-Alpha ControlNet REST API Server"
echo "=========================================="
echo "Image Size: ${IMAGE_SIZE}px"
echo "Config: $CONFIG"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "=========================================="
echo ""
echo "API Documentation: http://localhost:$PORT/docs"
echo ""

# Run the server
python app/api_server.py "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --image_size "$IMAGE_SIZE" \
    --port "$PORT"


