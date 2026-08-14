"""
PlanT2 agent for fail2drive evaluation.
Adapted from https://github.com/autonomousvision/plant2

Usage:
    PLANT_CHECKPOINT=./checkpoints/plant2/epoch=029_final_1.ckpt \
    SAVE_PATH=./eval_out \
    python leaderboard/leaderboard/leaderboard_evaluator_local.py \
      --routes ./fail2drive_split/Base_PedestriansOnRoad_0085.xml \
      --agent ./team_code/plant2_agent.py \
      --agent-config ./checkpoints/plant2 \
      --track MAP
"""

import os
import pathlib
import json
import re
from pathlib import Path
from datetime import datetime

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator

import carla
from data_agent import DataAgent
from plant2_lit_module import LitHFLM
from plant2_variables import PlanTVariables
from lateral_controller import LateralPIDController
from longitudinal_controller import LongitudinalLinearRegressionController
from config import GlobalConfig
import transfuser_utils as t_u

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'PlanT2Agent'


def strtobool(v):
    return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


def rad2deg(theta):
    return t_u.normalize_angle_degree(np.rad2deg(theta))


def generate_batch(data_batch):
    """Convert a list of samples into a batched tensor dict for the model."""
    maxseq = max([len(sample["input"]) for sample in data_batch])
    B = len(data_batch)

    x_batch_objs = [[0, 0, 0, 0, 0, 0, 0]]
    y_batch_objs = [[-999, -999, -999, -999]]

    batch_idxs = torch.zeros((B, maxseq), dtype=torch.int32)

    keys = [x for x in data_batch[0] if x not in ["input", "output"]]
    batches = {key: [] for key in keys}

    n = 1

    for i, sample in enumerate(data_batch):
        n_sample = len(sample["input"])
        batch_idxs[i, :n_sample] = torch.arange(n, n+n_sample)
        n += n_sample

        x_batch_objs.extend(sample["input"])
        y_batch_objs.extend(sample["output"])

        for key in keys:
            if key == "speed_limit":
                batches[key].append(torch.tensor(sample[key], dtype=torch.int))
            else:
                if torch.is_tensor(sample[key]):
                    batches[key].append(sample[key].type(torch.float32))
                else:
                    batches[key].append(torch.tensor(sample[key], dtype=torch.float32))

    batches = {key: torch.stack(value) for key, value in batches.items()}
    batches["idxs"] = batch_idxs
    batches["x_objs"] = torch.tensor(x_batch_objs, dtype=torch.float32)
    batches["y_objs"] = torch.tensor(y_batch_objs, dtype=torch.long)

    return batches


class PlanT2Agent(DataAgent):

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.route_index = route_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set TOWN/REPETITION for autopilot save_path logic
        if os.environ.get('SAVE_PATH') and not os.environ.get('TOWN'):
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            town = CarlaDataProvider.get_map().name.split('/')[-1]
            os.environ['TOWN'] = town
        if os.environ.get('SAVE_PATH') and not os.environ.get('REPETITION'):
            os.environ['REPETITION'] = '0'

        super().setup(path_to_conf_file, route_index, traffic_manager)

        # Load checkpoint
        ckpt_path = os.environ.get("PLANT_CHECKPOINT")
        if ckpt_path is None:
            ckpt_files = [f for f in os.listdir(path_to_conf_file) if f.endswith('.ckpt')]
            if ckpt_files:
                ckpt_path = os.path.join(path_to_conf_file, sorted(ckpt_files)[0])
            else:
                raise FileNotFoundError(f"No .ckpt file found in {path_to_conf_file}. "
                                        "Set PLANT_CHECKPOINT env var or place .ckpt in agent-config dir.")

        print(f'Loading PlanT2 from {ckpt_path}')

        # Extract config from checkpoint
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg_net = ckpt_data["hyper_parameters"]["cfg"]
        self.input_bev = self.cfg_net["model"]["training"].get("input_bev", False)
        self.input_static_cars = self.cfg_net["model"]["training"].get("input_static_cars", False)
        self.input_range = self.cfg_net["model"]["training"].get("range", 50)
        self.input_range_factor_front = self.cfg_net["model"]["training"].get("range_factor_front", 2)
        del ckpt_data

        print(f"  BEV: {self.input_bev}, Static cars: {self.input_static_cars}")
        print(f"  Range: {self.input_range}, Front factor: {self.input_range_factor_front}")

        # Patch HF checkpoint path to local if needed
        hf_ckpt = self.cfg_net["model"]["network"].get("hf_checkpoint", "prajjwal1/bert-medium")
        local_hf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "checkpoints", "hf_models", hf_ckpt)
        if os.path.exists(local_hf):
            self.cfg_net["model"]["network"]["hf_checkpoint"] = local_hf
            print(f"  Using local HF config: {local_hf}")

        self.net = LitHFLM.load_from_checkpoint(ckpt_path, map_location=self.device, strict=False)
        self.net.eval()

        # Controllers
        self.lat_pid = LateralPIDController(self.config)
        self.lon_pid = LongitudinalLinearRegressionController(self.config)

        self.plant_vars = PlanTVariables()
        self.speed_cats = self.plant_vars.speed_cats

        # Live visualization
        self.live_visu = strtobool(os.environ.get('LIVE_VISU', 'False'))
        self._visu_interface = None

        # XAI setup
        self.xai_enabled = int(os.environ.get('XAI_ENABLED', 0)) == 1
        if self.xai_enabled:
            print(f'PlanT2 XAI enabled: saving tensors to xai_tensors/')

        self.cleared_stop_sign = False
        self.moving_walkers = set()

        self.bev_colors = torch.tensor(self.plant_vars.bev_colors) if self.input_bev else None

    def sensors(self):
        result = [{
            "type": "sensor.other.imu",
            "x": 0.0, "y": 0.0, "z": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "sensor_tick": 0.05,
            "id": "imu"
        }, {
            "type": "sensor.speedometer",
            "reading_frequency": 20,
            "id": "speed"
        }, {
            'type': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'sensor_tick': 0.01,
            'id': 'gps'
        }]

        if self.live_visu:
            result.append({
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0],
                'y': self.config.camera_pos[1],
                'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0],
                'pitch': self.config.camera_rot_0[1],
                'yaw': self.config.camera_rot_0[2],
                'width': 1024,
                'height': 512,
                'fov': self.config.camera_fov,
                'id': 'rgb'
            })

        return result

    def tick(self, input_data):
        result = {}
        loc = self._vehicle.get_location()
        pos = np.array([loc.x, loc.y, loc.z])
        speed = input_data['speed'][1]['speed']
        compass = t_u.preprocess_compass(input_data['imu'][1][-1])

        result["speed"] = speed
        result["yaw"] = t_u.normalize_angle(compass)
        result['gps'] = pos[:2]

        if self.live_visu and 'rgb' in input_data:
            result["rgb"] = input_data["rgb"][1][:, :, :3]

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1

        if not self.initialized:
            self._init(None)

        tick_data = self.tick(input_data)

        # Route waypoints
        self._waypoint_planner.load()
        _, _, _, next_light_dist, next_traffic_light, next_stop_dist, next_stop_sign, speed_limit = \
            self._waypoint_planner.run_step(tick_data["gps"])
        waypoint_route = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index:][self.config.tf_first_checkpoint_distance:][::self.config.points_per_meter]
        waypoint_route = waypoint_route[:20, :2]
        self._waypoint_planner.save()

        tick_data["speed_limit"] = self.speed_cats.get(round(speed_limit*3.6), 0)
        tick_data["route"] = np.array([
            t_u.inverse_conversion_2d(p, tick_data['gps'], tick_data["yaw"])
            for p in waypoint_route
        ])

        # Pad route to 20 points if needed
        if len(tick_data["route"]) < 20:
            pad = np.tile(tick_data["route"][-1:], (20 - len(tick_data["route"]), 1))
            tick_data["route"] = np.concatenate([tick_data["route"], pad])

        # Bounding boxes
        label_raw = self.get_bounding_boxes()

        # Filter by range
        for x in label_raw:
            if "position" in x:
                pos_x, pos_y, pos_z = x["position"]
                x_div = self.input_range_factor_front**2 if pos_x > 0 else 1
                if pos_x**2/x_div + pos_y**2 > self.input_range**2 or abs(pos_z) > 30:
                    x["class"] = "too far"

        ego_vehicle_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)

        # Traffic lights
        if next_traffic_light is not None and next_light_dist < 30:
            for light, _, waypoints in self.list_traffic_lights:
                if light.id != next_traffic_light.id:
                    continue
                global_rot = light.get_transform().rotation
                relative_yaw = t_u.normalize_angle(np.deg2rad(global_rot.yaw) - ego_yaw)
                for wp in waypoints:
                    relative_pos = t_u.get_relative_transform(ego_matrix, np.array(wp.transform.get_matrix()))
                    label_raw.append({
                        'class': 'traffic_light',
                        'extent': [1.5, 1.5, 0.5],
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'state': str(light.state)
                    })

        # Stop sign
        if next_stop_sign is not None and not self.cleared_stop_sign and next_stop_dist < 30:
            center_bb = next_stop_sign.get_transform().transform(next_stop_sign.trigger_volume.location)
            stop_wp = self.world_map.get_waypoint(center_bb)
            rot = next_stop_sign.get_transform().rotation
            relative_yaw = t_u.normalize_angle(np.deg2rad(rot.yaw) - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, np.array(stop_wp.transform.get_matrix()))
            label_raw.append({
                'class': 'stop_sign',
                'extent': [1.5, 1.5, 0.5],
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw
            })

        # Stop sign clearing logic
        if next_stop_sign is not None:
            dist_to_stop = next_stop_sign.get_transform().transform(
                next_stop_sign.trigger_volume.location).distance(ego_vehicle_location)
        else:
            dist_to_stop = 999999

        if dist_to_stop > self.config.unclearing_distance_to_stop_sign:
            self.cleared_stop_sign = False
        elif self._vehicle.get_velocity().length() < 0.1 and dist_to_stop < self.config.clearing_distance_to_stop_sign:
            self.cleared_stop_sign = True

        # Get control
        control = self._get_control(label_raw, tick_data)

        # Initial brake frames
        if self.step < 40:
            control = carla.VehicleControl(0.0, 0.0, 1.0)

        # XAI: save tensors
        if self.xai_enabled and self.save_path is not None and self.step > self.config.xai_skip_first_steps:
            if self.step % self.config.xai_save_freq == 0:
                self._save_xai_tensors(label_raw, tick_data)

        # Live visualization + save to disk
        if self.live_visu and "rgb" in tick_data and tick_data["rgb"] is not None:
            self._display_live(tick_data["rgb"])
            if self.save_path is not None:
                self._save_visualization(tick_data["rgb"])

        return control

    def interpolate_waypoints(self, waypoints):
        """Interpolate waypoints to 0.1m spacing using PchipInterpolator."""
        waypoints = waypoints.copy()
        waypoints = np.concatenate((np.zeros_like(waypoints[:1]), waypoints))
        shift = np.roll(waypoints, 1, axis=0)
        shift[0] = shift[1]
        dists = np.linalg.norm(waypoints - shift, axis=1)
        dists = np.cumsum(dists)
        dists += np.arange(0, len(dists)) * 1e-4

        interp = PchipInterpolator(dists, waypoints, axis=0)
        x = np.arange(0.1, dists[-1], 0.1)
        interp_points = interp(x)

        if interp_points.shape[0] == 0:
            interp_points = waypoints[None, -1]

        return interp_points

    def _get_control(self, label_raw, input_data):
        gt_velocity = input_data['speed']
        input_batch = self._get_input_batch(label_raw, input_data)

        for x in input_batch:
            if input_batch[x] is not None:
                input_batch[x] = input_batch[x].to(self.device)

        input_batch["y_objs"] = None

        # Provide dummy BEV if model expects it but we don't have map data
        if self.input_bev and "BEV" not in input_batch:
            bev_size = self.config.lidar_resolution_width - 128
            input_batch["BEV"] = torch.zeros(1, 3, bev_size, bev_size, device=self.device)

        (pred_path, pred_wps, pred_speed) = self.net(input_batch)[2]

        if pred_path is not None:
            pred_path = pred_path.detach().squeeze().cpu().numpy()
        if pred_wps is not None:
            pred_wps = pred_wps.detach().squeeze().cpu().numpy()

        # Speed
        if pred_speed is not None:
            pred_speed = F.softmax(pred_speed.detach().squeeze().cpu(), dim=0).numpy()
            target_speeds = np.array(self.plant_vars.target_speeds)
            desired_speed = (target_speeds * pred_speed).sum()
        else:
            desired_speed = np.linalg.norm(pred_wps[2] - pred_wps[3]) * 4.0
            mean_speed = np.linalg.norm(pred_wps[:-1] - pred_wps[1:], axis=-1).mean() * 4.0
            if gt_velocity < 0.01:
                desired_speed = min(mean_speed, 0.1)

        throttle, brake = self.lon_pid.get_throttle_and_brake(desired_speed < 0.05, desired_speed, gt_velocity)

        # Steering
        if pred_path is not None:
            interp_wp = self.interpolate_waypoints(pred_path)
        else:
            interp_wp = self.interpolate_waypoints(pred_wps)

        if gt_velocity < 0.05 and brake:
            steer = self.lat_pid.step(
                np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
                gt_velocity, np.array([0., 0.]), 0., False)
        else:
            steer = self.lat_pid.step(interp_wp, gt_velocity, np.array([0., 0.]), 0., False)

        # Store for visualization
        self._last_pred_path = pred_path
        self._last_pred_wps = pred_wps
        self._last_desired_speed = desired_speed
        self._last_input_batch = input_batch
        self._last_gt_velocity = gt_velocity

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        return control

    def _get_input_batch(self, label_raw, input_data):
        """Convert raw labels + tick data into model input batch."""
        type_nums = dict(self.plant_vars.class_nums)
        car_types = self.plant_vars.car_types

        if not self.input_static_cars:
            type_nums.pop("static_car", None)

        # Filter walkers and classify objects
        for x in label_raw:
            if x.get("class") == "walker":
                if x.get("speed", 0) < 0.1 and x.get("id") not in self.moving_walkers:
                    x["class"] = "irrelevant_walker"
                else:
                    self.moving_walkers.add(x.get("id"))
            elif x.get("class") == "car" and x.get("type_id", "") in [
                    "vehicle.dodge.charger_police", "vehicle.dodge.charger_police_2020",
                    "vehicle.carlamotors.firetruck", "vehicle.ford.ambulance"]:
                x["class"] = "emergency"

        # Build feature vectors
        data_car = [
            [
                type_nums[x["class"].lower()],
                x['position'][0],
                x['position'][1],
                rad2deg(x['yaw']),
                x.get('speed', 0) * 3.6,
                x['extent'][1]*2,
                x['extent'][0]*2
            ]
            for x in label_raw
            if x.get("class", "").lower() in car_types
        ]

        data_car += [
            [
                type_nums[x["class"].lower()],
                x['position'][0],
                x['position'][1],
                rad2deg(x['yaw']),
                0.0,
                x['extent'][1]*2,
                x['extent'][0]*2
            ]
            for x in label_raw
            if x.get("class", "").lower() not in car_types
            and x.get("class", "").lower() in type_nums
            and (x.get("class", "").lower() != "traffic_light" or x.get("state") in ["Red", "Yellow"])
        ]

        sample = {
            'input': data_car,
            'output': [],
            'route_original': input_data["route"][:20],
            'speed_limit': input_data["speed_limit"],
            'ego_speed': input_data["speed"],
        }

        if self.input_bev and "BEV" in input_data:
            sample["BEV"] = input_data["BEV"]

        input_batch = generate_batch([sample])
        return input_batch

    def _save_xai_tensors(self, label_raw, tick_data):
        """Save model inputs for offline XAI analysis."""
        xai_dir = pathlib.Path(self.save_path) / 'xai_tensors'
        xai_dir.mkdir(parents=True, exist_ok=True)

        input_batch = self._get_input_batch(label_raw, tick_data)
        torch.save({
            'x_objs': input_batch['x_objs'].cpu(),
            'idxs': input_batch['idxs'].cpu(),
            'route_original': input_batch['route_original'].cpu(),
            'speed_limit': input_batch['speed_limit'].cpu(),
            'ego_speed': input_batch.get('ego_speed', torch.tensor(tick_data['speed'])).cpu(),
            'step': self.step,
            'model_type': 'plant2',
            'gt_speed': tick_data['speed'],
        }, xai_dir / f'{self.step:06d}.pt')

    def _save_visualization(self, rgb_image):
        """Save the combined RGB + BEV panel to disk (like TF++ does)."""
        rgb_h, rgb_w = rgb_image.shape[:2]
        bev_panel = self._render_synthetic_bev(rgb_w)
        rgb_bgr = rgb_image if rgb_image.shape[2] == 3 else rgb_image[:, :, :3]
        combined = np.concatenate([rgb_bgr, bev_panel], axis=0)
        cv2.imwrite(str(pathlib.Path(self.save_path) / f'{self.step:04d}.png'), combined)

    def _render_synthetic_bev(self, width):
        """Render a synthetic BEV showing object tokens, route, and predictions.

        Args:
            width: target width (to match RGB panel above)

        Returns:
            BGR image (width, width, 3) uint8
        """
        bev_size = width
        canvas = np.ones((bev_size, bev_size, 3), dtype=np.uint8) * 240

        # Coordinate system: ego at center, x=forward (up), y=left (left)
        origin = np.array([bev_size // 2, bev_size // 2])
        ppm = 8  # pixels per meter

        # Type colors (BGR): matches PlanTVariables class_nums
        type_colors = {
            1.0: (0, 165, 255),    # car: orange
            2.0: (0, 255, 0),      # walker: green
            3.0: (180, 180, 180),  # static: gray
            4.0: (160, 160, 250),  # stop_sign: pink
            5.0: (0, 0, 255),      # traffic_light: red
            6.0: (255, 0, 255),    # emergency: magenta
        }

        # Draw grid
        for i in range(0, bev_size, int(10 * ppm)):
            cv2.line(canvas, (i, 0), (i, bev_size), (220, 220, 220), 1)
            cv2.line(canvas, (0, i), (bev_size, i), (220, 220, 220), 1)

        # Draw range circle
        range_px = int(self.input_range * ppm)
        cv2.ellipse(canvas, tuple(origin), (range_px, int(range_px / self.input_range_factor_front)),
                    90, 0, 360, (200, 200, 200), 1)

        # Draw ego vehicle
        cv2.circle(canvas, tuple(origin), 8, (0, 180, 0), -1)
        cv2.arrowedLine(canvas, tuple(origin), (origin[0], origin[1] - 20), (0, 180, 0), 2)

        def world_to_px(x, y):
            """Convert ego-relative (x=forward, y=left) to pixel coords."""
            px_x = int(origin[0] + y * ppm)
            px_y = int(origin[1] - x * ppm)
            return (px_x, px_y)

        # Draw input objects from last batch
        if hasattr(self, '_last_input_batch') and self._last_input_batch is not None:
            x_objs = self._last_input_batch.get("x_objs")
            if x_objs is not None:
                x_objs_np = x_objs.cpu().numpy()
                for obj in x_objs_np[1:]:  # skip padding at index 0
                    obj_type = obj[0]
                    if obj_type == 0:
                        continue
                    x, y = obj[1], obj[2]
                    w_ext, l_ext = obj[5] / 2, obj[6] / 2

                    px = world_to_px(x, y)
                    color = type_colors.get(obj_type, (128, 128, 128))

                    # Draw as rectangle approximation
                    half_w = max(2, int(w_ext * ppm))
                    half_l = max(2, int(l_ext * ppm))
                    pt1 = (px[0] - half_w, px[1] - half_l)
                    pt2 = (px[0] + half_w, px[1] + half_l)
                    cv2.rectangle(canvas, pt1, pt2, color, -1)
                    cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), 1)

        # Draw route
        if hasattr(self, '_last_input_batch') and self._last_input_batch is not None:
            route = self._last_input_batch.get("route_original")
            if route is not None:
                route_np = route.cpu().numpy()[0]
                pts = [world_to_px(p[0], p[1]) for p in route_np if abs(p[0]) + abs(p[1]) > 0.01]
                if len(pts) > 1:
                    for i in range(len(pts) - 1):
                        cv2.line(canvas, pts[i], pts[i+1], (255, 180, 0), 2)

        # Draw predicted path (blue)
        if hasattr(self, '_last_pred_path') and self._last_pred_path is not None:
            pts = [world_to_px(p[0], p[1]) for p in self._last_pred_path]
            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    t = i / len(pts)
                    color = (int(255 * (1-t)), 50, int(255 * t))
                    cv2.line(canvas, pts[i], pts[i+1], color, 3)

        # Draw predicted waypoints (red circles)
        if hasattr(self, '_last_pred_wps') and self._last_pred_wps is not None:
            for i, wp in enumerate(self._last_pred_wps):
                px = world_to_px(wp[0], wp[1])
                t = i / max(1, len(self._last_pred_wps) - 1)
                color = (0, 0, int(128 + 127*t))
                cv2.circle(canvas, px, 5, color, -1)

        # Draw speed info
        if hasattr(self, '_last_desired_speed'):
            speed_text = f'Target: {self._last_desired_speed:.1f} m/s'
            cv2.putText(canvas, speed_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        if hasattr(self, '_last_gt_velocity'):
            gt_text = f'Current: {self._last_gt_velocity:.1f} m/s ({self._last_gt_velocity*3.6:.0f} km/h)'
            cv2.putText(canvas, gt_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Legend
        y_leg = bev_size - 120
        for type_val, color in type_colors.items():
            type_names = {1: 'Car', 2: 'Walker', 3: 'Static', 4: 'StopSign', 5: 'TrafLight', 6: 'Emergency'}
            name = type_names.get(int(type_val), '?')
            cv2.rectangle(canvas, (10, y_leg), (25, y_leg + 12), color, -1)
            cv2.putText(canvas, name, (30, y_leg + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y_leg += 18

        return canvas

    def _display_live(self, image):
        """Display combined RGB + synthetic BEV in a pygame window."""
        if image is None:
            return

        rgb_h, rgb_w = image.shape[:2]
        bev_panel = self._render_synthetic_bev(rgb_w)

        # Stack vertically: RGB on top, BEV on bottom
        rgb_bgr = image if image.shape[2] == 3 else image[:, :, :3]
        combined = np.concatenate([rgb_bgr, bev_panel], axis=0)

        if self._visu_interface is None:
            try:
                import pygame
                pygame.init()
                h, w = combined.shape[:2]
                self._visu_display = pygame.display.set_mode((w, h),
                                                             pygame.HWSURFACE | pygame.DOUBLEBUF)
                pygame.display.set_caption('PlanT2 Agent — RGB + BEV')
                self._visu_interface = pygame
            except ImportError:
                print('WARNING: pygame not installed, disabling live visualization')
                self.live_visu = False
                return

        for event in self._visu_interface.event.get():
            if event.type == self._visu_interface.QUIT:
                self.live_visu = False
                return
            if event.type == self._visu_interface.KEYDOWN and event.key == self._visu_interface.K_ESCAPE:
                self.live_visu = False
                return

        frame = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)
        surface = self._visu_interface.surfarray.make_surface(frame.swapaxes(0, 1))
        self._visu_display.blit(surface, (0, 0))
        self._visu_interface.display.flip()

    def _init(self, hd_map):
        super()._init(hd_map)
        self.initialized = True

    def destroy(self, results=None):
        if hasattr(self, '_visu_interface') and self._visu_interface is not None:
            self._visu_interface.quit()
        if hasattr(self, 'net'):
            del self.net
        super().destroy(results)