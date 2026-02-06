import logging
import math
import subprocess
from typing import Optional

from piper_sdk import C_PiperInterface_V2
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class PiperSdkAdapter:
    """
    Notes about piper_sdk units (from SDK docstrings):
      - Joint feedback (GetArmJointMsgs): 0.001 degrees :contentReference[oaicite:5]{index=5}
      - Joint command (JointCtrl): inputs are 0.001 degrees :contentReference[oaicite:6]{index=6}
      - High-speed feedback (GetArmHighSpdInfoMsgs):
          motor_speed: 0.001 rad/s
          effort: 0.001 N·m (doc says N/m but gripper doc says N·m; treat as torque-like) :contentReference[oaicite:7]{index=7}
    """

    def __init__(self, port: str):
        # port is CAN channel name like "can0"
        self.port = port
        self.can_setup_script = "./can_activate.sh"
        self._connected = False

        # Pass can_name explicitly (C_PiperInterface_V2 uses can_name internally) :contentReference[oaicite:8]{index=8}
        self.interface = C_PiperInterface_V2(can_name=port)

        # internal cache: last joint targets (deg), useful if you only command subset
        self._last_targets_deg: dict[int, float] = {i: 0.0 for i in range(1, 7)}

    def _ensure_can_ready(self):
        if self.can_setup_script:
            subprocess.run(["bash", self.can_setup_script], check=True)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return

        self._ensure_can_ready()

        # Start CAN read thread etc. :contentReference[oaicite:9]{index=9}
        self.interface.ConnectPort(can_init=False, piper_init=True, start_thread=True)

        # sanity check (will return a wrapper object; you can additionally validate Hz/time_stamp if you want)
        msg = self.interface.GetArmJointMsgs()  # :contentReference[oaicite:10]{index=10}
        if msg is None:
            raise ConnectionError("GetArmJointMsgs() returned None after ConnectPort()")

        self._connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.__class__.__name__}('{self.port}') is not connected.")

        if disable_torque:
            try:
                self.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")

        # Disable + stop threads/close CAN
        self.interface.DisablePiper()  # :contentReference[oaicite:11]{index=11}
        self.interface.DisconnectPort()

        self._connected = False

    # ---- torque(enable/disable) ----
    def enable_torque(self, joint_ids: Optional[list[int]] = None) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # SDK: EnableArm motor_num 1..7, 7=all motors :contentReference[oaicite:12]{index=12}
        if joint_ids is None:
            self.interface.EnableArm(7)
            return

        for jid in joint_ids:
            if jid == 7:
                self.interface.EnableArm(7)
            else:
                self.interface.EnableArm(int(jid))

    def disable_torque(self, joint_ids: Optional[list[int]] = None) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # SDK: DisableArm motor_num 1..7, 7=all motors :contentReference[oaicite:13]{index=13}
        if joint_ids is None:
            self.interface.DisableArm(7)
            return

        for jid in joint_ids:
            if jid == 7:
                self.interface.DisableArm(7)
            else:
                self.interface.DisableArm(int(jid))

    # ---- read ----
    def read_joint_positions_deg(self, joint_ids: list[int]) -> dict[int, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        msg = self.interface.GetArmJointMsgs()
        if msg is None:
            raise RuntimeError("GetArmJointMsgs() returned None")

        js = msg.joint_state  # ArmMsgFeedBackJointStates :contentReference[oaicite:14]{index=14}
        all_deg = {
            1: js.joint_1 * 0.001,
            2: js.joint_2 * 0.001,
            3: js.joint_3 * 0.001,
            4: js.joint_4 * 0.001,
            5: js.joint_5 * 0.001,
            6: js.joint_6 * 0.001,
        }
        return {jid: all_deg[jid] for jid in joint_ids}

    def read_joint_velocities_deg_s(self, joint_ids: list[int]) -> dict[int, float]:
        """
        Uses high-speed motor feedback:
          motor_speed is 0.001 rad/s -> deg/s = motor_speed*0.001*180/pi :contentReference[oaicite:15]{index=15}
        If unavailable, returns 0.0 for those joints.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        info = self.interface.GetArmHighSpdInfoMsgs()
        if info is None:
            return {jid: 0.0 for jid in joint_ids}

        # info.motor_1 ... info.motor_6
        motor = {
            1: info.motor_1,
            2: info.motor_2,
            3: info.motor_3,
            4: info.motor_4,
            5: info.motor_5,
            6: info.motor_6,
        }

        out: dict[int, float] = {}
        for jid in joint_ids:
            ms = getattr(motor[jid], "motor_speed", None)
            if ms is None:
                out[jid] = 0.0
            else:
                out[jid] = (ms * 0.001) * (180.0 / math.pi)
        return out

    def read_joint_torques(self, joint_ids: list[int]) -> dict[int, float]:
        """
        Uses high-speed motor feedback:
          effort is 0.001 (torque-like) -> Nm approx = effort*0.001 :contentReference[oaicite:16]{index=16}
        If unavailable, returns 0.0.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        info = self.interface.GetArmHighSpdInfoMsgs()
        if info is None:
            return {jid: 0.0 for jid in joint_ids}

        motor = {
            1: info.motor_1,
            2: info.motor_2,
            3: info.motor_3,
            4: info.motor_4,
            5: info.motor_5,
            6: info.motor_6,
        }

        out: dict[int, float] = {}
        for jid in joint_ids:
            eff = getattr(motor[jid], "effort", None)
            out[jid] = 0.0 if eff is None else (eff * 0.001)
        return out

    # ---- write ----
    def send_joint_positions_deg(self, targets: dict[int, float]) -> None:
        """
        targets: {joint_id: target_deg}
        Internally sends JointCtrl(j1..j6) with unit 0.001 degrees :contentReference[oaicite:17]{index=17}
        """
        if not self.is_connected:
            raise DeviceNotConnectedError("Device not connected")

        # update cache
        for jid, deg in targets.items():
            if jid < 1 or jid > 6:
                raise ValueError(f"joint_id must be 1..6, got {jid}")
            self._last_targets_deg[jid] = float(deg)

        def to_mdeg(deg: float) -> int:
            # 0.001 degree unit (milli-degree)
            return int(round(deg * 1000.0))

        j1 = to_mdeg(self._last_targets_deg[1])
        j2 = to_mdeg(self._last_targets_deg[2])
        j3 = to_mdeg(self._last_targets_deg[3])
        j4 = to_mdeg(self._last_targets_deg[4])
        j5 = to_mdeg(self._last_targets_deg[5])
        j6 = to_mdeg(self._last_targets_deg[6])

        self.interface.JointCtrl(j1, j2, j3, j4, j5, j6)  # :contentReference[oaicite:18]{index=18}
