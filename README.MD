### Citation
**This project uses code from the mink project by Kevin Zakka (https://github.com/kevinzakka/mink), licensed under Apache-2.0.**
``` bibtex
@software{Zakka_Mink_Python_inverse_2026,
  author  = {Zakka, Kevin},
  title   = {{Mink: Python inverse kinematics based on MuJoCo}},
  year    = {2026},
  month   = march,
  version = {1.1.0},
  url     = {https://github.com/kevinzakka/mink},
  license = {Apache-2.0}
}
```
### make conda env
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
```

### dependencies
```bash
pip install -r requirements.txt
```

### quest3 setting
Quest3 WebXR and WebSocket-based teleoperation requires a secure HTTPS connection. Modern browsers block access to sensors (WebXR, camera, motion tracking, etc.) when the page is served over plain HTTP. Therefore, a local SSL certificate must be generated to run the teleoperation server securely.

The following commands generate a trusted local certificate using mkcert.
```bash
cd ~/quest3
mkcert -install
mkcert -cert-file cert.pem -key-file key.pem 192.168.xxx.xxx localhost 127.0.0.1
```
**server url**
Access the server from Quest3 using:
https://192.168.xxx.xxx:8012?ws=wss://192.168.xxx.xxx:8012

### /description
Robot description files

### /mink_ik
This directory is used to test IK control with Mink. It also supports teleoperation with Quest3.

Run the following command to test Agilex Piper:
```bash
python3 -m mink_ik.single_arm_mink_ik
```

### /quest3
This directory contains the server that receives controller inputs from Quest3 and preprocesses them for control.

Note that Vuer uses a coordinate convention of Y-up, -Z forward, while MuJoCo uses Z-up, X-forward.

### /lerobot_mujoco
This directory contains LeRobot-based code for the MuJoCo environment.
#### 1. data collection
```bash
python3 -m lerobot_mujoco.single_arm.data_collection
```
#### 2. data visualization
```bash
python3 -m lerobot_mujoco.single_arm.data_visualization
```
#### 3. training
Run the following command to train the ACT policy using a collected LeRobot dataset:
```bash
python train_act.py \
  --root ./demo_data \
  --repo_id single_arm_teleop \
  --out ./ckpt/act_single_arm \
  --chunk_size 10 \
  --n_action_steps 10 \
  --batch_size 64 \
  --num_workers 4 \
  --lr 1e-4 \
  --steps 5000 \
  --device cuda```
```
#### 4. deploy
```bash
python3 -m lerobot_mujoco.single_arm.deploy
```

### /lerobot_real
This directory contains LeRobot-based code for the real robot environment.
Currently, Agilex Piper is supported.

Its overall workflow is similar to `/lerobot_mujoco`
