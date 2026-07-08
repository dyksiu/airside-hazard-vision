# Airside Hazard Vision

**MSc Thesis — Evaluation of Computer Vision Algorithms for Driver Support and Hazard Warning in Airport Airside Environments**

This repository contains the implementation of a computer vision system developed as part of an MSc thesis. The system supports real-time video analysis for vehicles operating on airport aprons, taxiways and other airside surfaces.

The project focuses on surface recognition and horizontal ground-marking detection using deep learning models. It compares several computer vision approaches and evaluates their usability in a driver support and hazard warning system for airport airside environments.

## Demo

### Example operation of the driver-support system during airport airside video analysis.

<img width="1152" height="648" alt="Application demo" src="https://github.com/user-attachments/assets/0343112a-a3aa-40ae-9724-cdb249d33ea4" />

### Main application window

<img width="1449" height="751" alt="GUI_1" src="https://github.com/user-attachments/assets/8028ea6b-4a4a-46fa-94dd-648d2ff5bd84" />

### ROI comparison

Comparison of full-frame analysis and ROI-based analysis.

| ROI-based analysis | Full-frame analysis |
|---|---|
| <img width="485" alt="ROI-based analysis" src="https://github.com/user-attachments/assets/dcb0616b-82bd-4fdb-85a5-5ae631383430" /> | <img width="486" alt="Full-frame analysis" src="https://github.com/user-attachments/assets/825c723e-4be0-4d54-941d-4f4f703f8764" /> |


## Overview

Airside environments require fast and reliable interpretation of surface markings, infrastructure elements and operational zones. Ground support vehicles often operate close to aircraft, service equipment and restricted areas, where crossing a marked boundary may create a safety risk.

The goal of this project is to analyse and compare computer vision algorithms that can support a driver of an airport vehicle by recognising:

- airport surface types,
- horizontal ground markings,
- safety-related markings and restricted zones,
- visual cues useful for warning and decision support.

The system processes video footage, runs selected deep learning models and presents the results in a graphical application. It can visualise detected classes, segmentation masks, processing speed, warning events and detection statistics.

## Main features

- Real-time video processing from recorded airport airside footage
- Semantic segmentation of airport surfaces
- Detection and segmentation of horizontal ground markings
- Support for multiple model families:
  - YOLO
  - U-Net
  - DeepLabV3
  - SegFormer
- Graphical desktop application
- Model selection from the user interface
- Confidence threshold configuration
- Optional ROI-based analysis of the lower part of the image
- Visualisation of masks, labels and warning information
- Current and average FPS monitoring
- Event log for warnings and detections
- Tabular preview of detection results
- CSV export of analysis results
- Experimental embedded deployment on Raspberry Pi 5 with Raspberry Pi AI HAT / Hailo accelerator

## Repository structure

```text
airside-hazard-vision/
├── core/                  # Core video processing and analysis logic
├── models/                # Model-related files and utilities
├── training and mapping/  # Training, annotation mapping and dataset preparation scripts
├── ui/                    # Graphical user interface components
├── assets/                # Recommended folder for screenshots and demo GIFs
├── app.py                 # Application-related entry or support file
├── main.py                # Main application entry point
├── make_requirements.py   # Utility for requirements generation
├── requirements.txt       # Python dependencies
└── README.md
```

## Technologies

The project uses Python and common computer vision / deep learning libraries:

- Python
- OpenCV
- NumPy
- Pillow
- Matplotlib
- PyTorch
- Torchvision
- Ultralytics
- Transformers
- Roboflow

## Models evaluated

The project compares several deep learning approaches for airport scene analysis.

| Model family | Main task |
|---|---|
| YOLO | Fast object detection and segmentation |
| U-Net | Semantic segmentation |
| DeepLabV3 | Semantic segmentation with atrous convolution and ASPP |
| SegFormer | Transformer-based semantic segmentation |

The comparison considers both prediction quality and practical usability in a driver-support system, including processing speed, model stability and deployment constraints.

## Dataset

The datasets were prepared from video recordings captured in an airport airside environment. Frames were extracted from video material, selected, annotated and split into training, validation and test subsets.

Separate datasets were prepared for:

- airport surface segmentation,
- horizontal ground-marking segmentation.

Data augmentation was applied only to the training subset.

> Note: datasets, trained weights and video materials may be stored externally because of file size, licensing or access restrictions.

## Installation

Clone the repository:

```bash
git clone https://github.com/dyksiu/airside-hazard-vision.git
cd airside-hazard-vision
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a PyTorch version compatible with your CUDA setup.

You can verify CUDA availability with:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Running the application

Start the application from the main project directory:

```bash
python main.py
```

After launching the GUI:

1. Select the model for surface analysis.
2. Select the model for horizontal marking analysis.
3. Choose an input video file.
4. Set the confidence threshold.
5. Optionally configure the ROI parameter.
6. Start processing.

The application displays the processed video with detected classes and masks. It also shows current and average FPS, event logs and detection results. Results can be exported to a CSV file.

## Windows executable

A ready-to-use Windows executable is available in the release section:

```text
https://github.com/dyksiu/airside-hazard-vision/releases/tag/v1.0.0
```

This version can be used without manually running the Python source code or installing all dependencies.

## Raspberry Pi / Hailo version

An embedded version was prepared for Raspberry Pi 5 with Raspberry Pi AI HAT / AI HAT+ and a Hailo accelerator.

Clone the embedded branch:

```bash
git clone -b embedded_rpi_hailo https://github.com/dyksiu/airside-hazard-vision.git
cd airside-hazard-vision
```

Create and activate the environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Hailo-related packages on Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y dkms
sudo apt install -y hailo-all
sudo reboot
```

Check whether the Hailo accelerator is detected:

```bash
lspci | grep -i Hailo
hailortcli fw-control identify
dmesg | grep -i hailo
ls /dev/hailo*
```

Run the application:

```bash
python main.py
```

The Raspberry Pi version requires models prepared in a format supported by the Hailo accelerator, for example `.hef` files.

## Results summary

The experiments compared YOLO, U-Net, DeepLabV3 and SegFormer models in terms of prediction quality and processing speed.

In general:

- YOLO-based configurations were better suited for real-time processing.
- SegFormer achieved strong prediction confidence, especially for surface and marking recognition.
- U-Net, DeepLabV3 and SegFormer provided useful semantic segmentation results, but with lower processing speed compared to lightweight YOLO configurations.
- Raspberry Pi / Hailo tests confirmed the feasibility of embedded deployment, while also showing that GUI rendering can significantly affect overall performance.

The final model choice depends on the system priority:

- maximum segmentation quality,
- stable real-time performance,
- embedded deployment feasibility.

## Possible future improvements

Potential directions for further development include:

- expanding the dataset with more weather and lighting conditions,
- improving visualisation of warning events,
- optimising model inference for embedded platforms,
- testing additional lightweight segmentation models,
- adding support for live camera input,
- improving the user interface for operational testing.

## Academic context

This repository accompanies the MSc thesis:

**Evaluation of Computer Vision Algorithms for Driver Support and Hazard Warning in Airport Airside Environments**

Author: **Maciej Dyks**  
Poznań University of Technology, 2026
