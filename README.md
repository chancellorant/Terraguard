 Terraguard
 Wildfire Detection System
TerraGuard is a ground-based, multi-sensor wildfire detection system designed to detect incipient ignition using environmental sensing and edge AI vision fusion.
Developed by Anant Arora
North High School – Torrance, California

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


