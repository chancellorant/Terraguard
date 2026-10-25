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


