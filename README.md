 # TerraGuard

### Ground-Based Pre-Visible Wildfire Detection Using Multi-Sensor Fusion and Edge Intelligence

Developed by Anant Arora  
North High School | Torrance, California

<img src="docs/images/Terraguard outdoor field Deployment.png" alt="TerraGuard outdoor field deployment" width="650">

*TerraGuard ground-based monitoring system during outdoor field testing.*

## Overview

TerraGuard is an independent research project investigating whether combustion-related environmental changes near the ground can provide an earlier indication of possible wildfire ignition before reliable visual confirmation is available.

The system follows a **disturbance-first, multi-sensor fusion approach**. Ground-level sensors continuously monitor volatile organic compound (VOC) trends, fine particulate matter (PM₂.₅), temperature, humidity, and atmospheric conditions. When sustained environmental deviations are detected, visual and thermal sensing provide independent evidence to evaluate whether a combustion event may be occurring.

TerraGuard integrates four coordinated subsystems: **TerraSense** (environmental sensing), **TerraVision** (visual and thermal detection), **TerraGuard Command Center** (edge-based processing and multi-sensor fusion), and **TerraNet** (wireless alert communication).

Controlled indoor experiments and structured outdoor trials examine early combustion detection, system response, false-positive suppression, and the reliability of multi-sensor fusion under varying environmental conditions.



## System Architecture

TerraGuard integrates ground-level environmental sensing, AI-assisted visual detection, thermal imaging, edge-based intelligence, and wireless communication within a coordinated wildfire detection architecture.

```text
        TerraSense                    TerraVision
    Environmental Sensing        Visual & Thermal Detection
    VOC • PM₂.₅ • Microclimate       RGB • Thermal • AI
             |                              |
             |                              |
             +--------------+---------------+
                            |
                            v
               TerraGuard Command Center
               Edge Intelligence & Fusion
                            |
                            v
                         TerraNet
                 Wireless Communication
                            |
                            v
                 Gateway & Dashboard
                      Incident Alerts
```

### Core Subsystems

| Subsystem | Primary Function |
|---|---|
| **TerraSense** | Monitors near-ground environmental conditions and identifies sustained combustion-related disturbances. |
| **TerraVision** | Provides AI-assisted visual detection and thermal evidence for confirmation of suspected combustion events. |
| **TerraGuard Command Center** | Integrates environmental, visual, and thermal information through on-device processing and multi-sensor fusion to determine alert status. |
| **TerraNet** | Transmits confirmed incident information to a gateway for delivery to monitoring dashboards and connected interfaces. |

### Detection Workflow

Environmental Monitoring  
↓  
Disturbance Identification  
↓  
Visual/Thermal Evaluation  
↓  
Multi-Sensor Fusion  
↓  
Alert Generation  
↓  
Wireless Notification

Environmental sensing provides the initial indication of a potential combustion event. Visual and thermal systems provide additional evidence, while edge-based processing coordinates multi-sensor fusion, alert validation, and wireless notification.


## Hardware

TerraGuard integrates environmental sensing, visual and thermal imaging, edge computing, wireless communication, and power subsystems within a ground-based wildfire monitoring system.

| Component | Function |
|---|---|
| **Raspberry Pi 5 (8 GB)** | Central edge computer for sensor integration, local processing, and multi-sensor fusion. |
| **Hailo-8L AI Accelerator (13 TOPS)** | Accelerates on-device AI inference for visual fire and smoke detection. |
| **Bosch BME688** | Measures gas resistance as a proxy for VOC trends, along with temperature, humidity, and atmospheric pressure. |
| **PMS5003** | Measures particulate matter (PM₁, PM₂.₅, and PM₁₀). |
| **ADS1115 Analog-to-Digital Converter** | Converts analog sensor signals for processing by the Raspberry Pi. |
| **Reolink 4K PTZ Camera** | Provides RGB imagery for AI-assisted fire and smoke detection. |
| **FLIR Lepton 3.5** | Provides thermal imaging for independent heat-related evidence. |
| **Wind Vane and Anemometer** | Measure wind speed and direction to support environmental analysis and field testing. |
| **GPS Module** | Provides geographic location information for the monitoring node. |
| **LILYGO T-Beam (915 MHz LoRa)** | Supports wireless transmission of confirmed incident alerts to a gateway. |
| **Telit LE910 LTE Modem / GL.iNet Router** | Supports cellular network connectivity, remote access, and dashboard communication. |
| **Portable Power Station (288 Wh)** | Supplied electrical power during experimental testing. |
| **Solar Panel (150 W) and 12 V LiFePO₄ Battery System** | Designed for future off-grid operation; autonomous solar-powered performance was not validated in the reported experiments. |



## Experimental Data

TerraGuard's experimental research includes controlled indoor combustion tests, structured outdoor trials, and AI-based fire and smoke detection analysis. The collected datasets document environmental sensor responses, detection behavior, and system performance under different experimental conditions.

The repository contains three experimental data archives:

| Data Archive | Description |
|---|---|
| **INDOORCSV.zip** | Environmental sensor measurements and detection records collected during controlled indoor combustion experiments. |
| **CSV OUTDOOR.zip** | Environmental sensor measurements and alert records collected during structured outdoor experiments, including trials conducted at distances of 2 m, 4 m, and 8 m. |
| **Ai csv.zip** | YOLOv8 model training and evaluation records, including precision, recall, and mAP metrics, together with NASA FIRMS satellite fire-detection reference data. |

These datasets accompany the experimental findings documented in the TerraGuard research paper and provide supporting records for the investigation of early combustion indicators, multi-sensor fusion, and edge-based wildfire detection.


## Limitations

The experimental findings reflect controlled indoor and structured outdoor testing. The research identifies several limitations that require further investigation:

Environmental transport and detection range: Combustion-related gases and particles undergo transport, mixing, and dilution before reaching ground-level sensors. Detection performance depends on distance, wind conditions, and sensor placement. The experiments examined distances of 2–8 m; performance over larger areas remains to be established.

Experimental scale and sample size: Testing used small combustion sources, including candle flames and wood fragments. The study included three repetitions per indoor scenario, eight non-combustion trials, and two systematically recorded outdoor trials. Larger datasets and additional field experiments are needed to evaluate performance across more representative wildfire conditions.

Environmental interference and model generalization: Non-fire atmospheric disturbances may produce sensor responses that overlap with combustion-related signals. Visual detection performance may also vary under heavy smoke, occlusion, low illumination, and other conditions not fully represented in the training and testing datasets.

Communication and operational reliability: LoRa communication was tested over 2–8 m line-of-sight distances. Extended-range communication, multi-node network performance, and long-duration system stability require further evaluation. Intermittent service interruptions were observed during extended operation, and formal packet delivery rate measurements were not completed.

Thermal confirmation and system scalability: The thermal confirmation threshold requires formal characterization and calibration. Broader node deployment, network coordination, and integration with operational fire-management systems remain subjects for future validation.



## Repository Structure

The repository contains TerraGuard's subsystem software, experimental datasets, and supporting documentation.

```text
.
├── README.md
├── .gitattributes
├── terrasense/
├── terraguardcommandcenter/
├── terranet/
├── docs/
│   └── images/
├── INDOORCSV.zip
├── CSV OUTDOOR.zip
└── Ai csv.zip
```

### Repository Contents

| Directory or File | Description |
|---|---|
| `terrasense/` | Environmental sensor acquisition and monitoring code. |
| `terraguardcommandcenter/` | Visual detection, edge processing, and monitoring dashboard code. |
| `terranet/` | Communication and network connectivity code, including LTE modem configuration. |
| `docs/images/` | Photographs and supporting project documentation. |
| `INDOORCSV.zip` | Controlled indoor experimental data. |
| `CSV OUTDOOR.zip` | Structured outdoor experimental data. |
| `Ai csv.zip` | AI model training and evaluation records, together with NASA FIRMS satellite reference data. |





## Current Research Direction

TerraGuard's ongoing research focuses on understanding how combustion-related environmental signatures propagate through the atmosphere before reaching ground-level sensors.

The central research question is:

> How do distance, wind speed, and wind direction affect the transport, dilution, and detection timing of early combustion signatures such as VOCs and PM₂.₅ prior to reliable visual confirmation?

Building on the initial indoor and outdoor experiments, the next phase of investigation focuses on:

- Examining how combustion-related sensor responses change with increasing distance from the ignition source.
- Investigating how wind speed and wind direction influence the transport and dilution of combustion-related gases and fine particles.
- Evaluating how environmental conditions affect detection timing, consistency, and reliability.

The objective is to improve understanding of atmospheric transport processes that influence ground-level detection and establish an experimental foundation for future wildfire detection research.

## Getting Started

The following steps describe how to configure and run the TerraGuard software stack on a Raspberry Pi.

### 1. Install Python Dependencies

Install the required Python libraries on the Raspberry Pi:

```bash
pip install flask numpy opencv-python requests pyserial \
  adafruit-blinka adafruit-circuitpython-bme680 \
  adafruit-circuitpython-ads1x15 tflite-runtime
```

### 2. Configure Application Settings

Create a local configuration file from the provided template:

```bash
cp terraguardcommandcenter/config_example.py terraguardcommandcenter/config.py
```

Edit `config.py` to specify your camera address and trained model file path. Keep this file private and do not upload it to GitHub.

### 3. Start TerraSense

Launch the environmental sensor dashboard:

```bash
python3 terrasense/terrasensedashboard.py
```

### 4. Start RGB Detection

Launch the RGB inference server on port 5000:

```bash
cd terraguardcommandcenter
python3 unifieddashboard.py
```

### 5. Start TerraGuard Command Center

From the `terraguardcommandcenter/` directory, launch the Command Center dashboard on port 8092:

```bash
RGB_BASE="http://127.0.0.1:5000" PORT="8092" python3 terraguarddashboard.py
```

### 6. Access the Dashboard


Open a browser on the same network and navigate to:

```text
http://<raspberry-pi-address>:8092
```
## Author

Anant Arora  
North High School  
Torrance, California  
[LinkedIn](https://www.linkedin.com/in/anant-arora-0aaa39383)


## Acknowledgments

- NASA FIRMS (Fire Information for Resource Management System) for satellite fire-detection reference data.
- D-Fire dataset contributors for publicly available fire and smoke imagery used in AI model training and evaluation.
- CAL FIRE Fire Innovation Unit for discussion of the TerraGuard research during an October 2026 presentation.
- Professor Michael Gollner, University of California, Berkeley, for scientific feedback concerning atmospheric transport and the detection of early combustion signatures.




