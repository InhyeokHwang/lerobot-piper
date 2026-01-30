from piper_sdk import C_PiperInterface_V2

class PiperBus:
    def __init__(self, config):
        self.config = config
        self.arm = None

        self.joint_names = [f"joint_{i}" for i in range(1, config.dof + 1)] # 1~6

    def connect(self):
        
    def disconnect(self):

    def read_joint_state(self):


    def send_joint_target(self, target):

    
    def read_gripper(self):

    
    def send_gripper(self, target):

    