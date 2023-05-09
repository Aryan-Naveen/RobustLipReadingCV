echo =========================================== Preprocessing Videos ===========================================;
python facial_tracking/tracking_main.py;

echo =========================================== Performing Predictions ===========================================;
cd deep_lip_reading;
python main.py --lip_model_path models/lrs2_lip_model;
cd ..;
