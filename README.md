# direct-PINN
Use PINN to predict data.

## Usage
Install dependencies:
```bash
pip install numpy>=1.21.0 torch>=1.10.0 meshio>=5.0.0
```

Train on the first 601 VTU files and predict the next 200 time steps:
```bash
python pinn_predict.py --data-dir /path/to/vtu_folder --train-steps 601 --predict-steps 200 --nu 0.01
```

If your VTU point_data keys differ, specify them:
```bash
python pinn_predict.py --data-dir /path/to/vtu_folder --velocity-key Velocity --pressure-key Pressure
```
