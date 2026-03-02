### quest3 setting
cd ~/quest3
mkcert -install

mkcert -cert-file cert.pem -key-file key.pem \
192.168.50.172 localhost 127.0.0.1

server url
https://192.168.xxx.xxx:8012?ws=wss://192.168.xxx.xxx:8012



train 
python3 -m lerobot_mujoco.single_arm.train \
  --root /home/incheol/lerobot-robot-piper/lerobot_mujoco/single_arm/demo_data \
  --repo_id single_arm_teleop \
  --out /home/incheol/lerobot-robot-piper/ckpt/act_single_arm \
  --chunk_size 10 \
  --n_action_steps 10 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 1e-4 \
  --steps 1000 \
  --device cuda


deploy
python3 -m lerobot_mujoco.single_arm.deploy \
  --ckpt ./ckpt/act_single_arm \
  --repo_id piper_single_arm_teleop \
  --data_root ./lerobot_mujoco/single_arm/demo_data \
  --ee_site gripper \
  --hz 20