# Interaction-to-Image Backend

Backend server for image generation with interactive feedback.

## Setup

### 1. Create Conda Environment

```bash
cd /home/ella/courses/HCI/interaction-to-image
conda env create -f i2i/environment.yml
conda activate hci_i2i
```

### 2. Install Additional Packages

```bash
cd back
pip install -r requirements.txt
```

### 3. Model Download

Models are automatically downloaded from HuggingFace on first run.

## Running the Server

### Method 1: Direct Execution (Recommended)

```bash
cd /home/ella/courses/HCI/interaction-to-image/back
conda activate hci_i2i
python main.py
```

### Method 2: Run as Package

```bash
cd /home/ella/courses/HCI/interaction-to-image
conda activate hci_i2i
python -m back.main
```

## API Endpoints

- `POST /api/composition/objects` - Create object list from prompt
- `POST /api/composition/start` - Start image generation session
- `POST /api/branch/create` - Create branch with feedback
- `WebSocket /ws/image-stream/{session_id}` - Image streaming
- `WebSocket /ws/image-stream/{session_id}/{branch_id}` - Branch image streaming

## Server Architecture

- **HTTP Server**: Port 8000 (handles REST API)
- **WebSocket Server**: Port 8001 (handles image streaming)

For detailed API specification, see `BACKEND_SPEC.md` in the root directory.
