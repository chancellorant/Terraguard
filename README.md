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

