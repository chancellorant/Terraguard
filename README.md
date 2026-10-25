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
Project Structure
Terraguard command center/
•	Multi-sensor fusion logic
•	Edge AI inference
•	Alert decision engine
terrasense/
•	Environmental sensor acquisition (VOC, PM, temperature, humidity)
•	Baseline modeling
•	Suspicion state detection
terranet/
•	LoRa mesh communication module
•	Alert packet transmission

Experimental Data (CSV Logs)
INDOORCSV.zip
•	Indoor controlled combustion trials
•	Includes VOC, PM2.5, temperature, YOLO confidence, fusion state logs
CSV OUTDOOR.zip
•	Outdoor structured ignition trials
•	Includes distance-based detection data (2m, 4m, 8m)
AI csv.zip
•	Vision model inference outputs
•	Bounding box confidence logs

System Capabilities
•	Disturbance-first detection (VOC + PM2.5)
•	YOLOv8 fire/smoke detection (Hailo-8L accelerated)
•	Multi-modal fusion logic
•	Edge-based inference (<5s alert latency)
•	LoRa mesh communication


