# direct-PINN
Use PINN to predict data.

## Usage
Install dependencies:
`pip install numpy torch meshio`

Train on the first 601 VTU files and predict the next 200 time steps:
`python pinn_predict.py --data-dir /path/to/vtu_folder --train-steps 601 --predict-steps 200 --nu 0.01`

If your VTU point_data keys differ, specify them:
`python pinn_predict.py --data-dir /path/to/vtu_folder --velocity-key Velocity --pressure-key Pressure`
