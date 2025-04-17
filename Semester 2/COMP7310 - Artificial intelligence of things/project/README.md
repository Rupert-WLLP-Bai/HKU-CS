# AIOT Project

## backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server will be running on `http://localhost:5000`.

## esp32

### csi_recv

1. add `mqtt` to `CMakeLists.txt` in `csi_recv/main` folder
2. run `idf.py add-dependency esp-dsp` to add `esp-dsp` dependency which will generate `managed_components` folder.
3. build and flash (WiFi configuration needs to be modified according to your network)

## csi_send

1. build and flash (WiFi configuration needs to be modified according to your network)

## onboard

1. run `breathing_estimation_dataset.py` to generate results and graphs, which will be saved in `benchmark/breathing_rate/evaluation/visualization` folder
2. run `breathing_estimation_testset.py` to generate results.
3. open `motion_detection.ipynb` and run to generate results and graphs