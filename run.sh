echo =========================================== Cropping Deblurred Faces ===========================================;
conda activate lipReadEnv;
python facial_tracking/tracking_main.py 0;

echo =========================================== Deblurring Video ===========================================;
conda activate simdeblur;
cd deblur;
python deblur.py
cd ..;

echo =========================================== Performing Predictions ===========================================;
conda activate lipReadEnv;
cd deep_lip_reading;
python main.py --lip_model_path models/lrs2_lip_model;
cd ..;

# echo =========================================== Cropping Blurred Faces ===========================================;
# conda activate lipReadEnv;
# python facial_tracking/tracking_main.py 1;
#
# echo =========================================== Performing Predictions ===========================================;
# cd deep_lip_reading;
# python main.py --lip_model_path models/lrs2_lip_model;
# cd ..;
