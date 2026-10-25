 # TerraGuard

### Ground-Based Pre-Visible Wildfire Detection Using Multi-Sensor Fusion and Edge Intelligence

Developed by Anant Arora  
North High School | Torrance, California

<img src="docs/images/Terraguard outdoor field Deployment.png" alt="TerraGuard outdoor field deployment" width="650">

*TerraGuard ground-based monitoring system during outdoor field testing.*

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


