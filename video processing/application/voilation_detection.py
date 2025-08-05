import cv2
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import torch
import torchvision.transforms as transforms
from collections import defaultdict, deque
import math

class IndianTrafficViolationDetector:
    """
    Comprehensive traffic violation detection system for Indian traffic conditions
    """

    def __init__(self):
        self.violation_types = {
            # Speed violations
            'overspeeding': {'speed_limit': 60, 'confidence_threshold': 0.8},
            'rash_driving': {'acceleration_threshold': 5.0, 'confidence_threshold': 0.75},

            # Lane violations
            'lane_cutting': {'confidence_threshold': 0.85},
            'wrong_lane': {'confidence_threshold': 0.80},
            'lane_discipline_violation': {'confidence_threshold': 0.75},

            # Signal violations
            'red_light_jumping': {'confidence_threshold': 0.90},
            'signal_violation': {'confidence_threshold': 0.85},

            # Safety violations
            'no_helmet': {'confidence_threshold': 0.85},
            'no_seatbelt': {'confidence_threshold': 0.80},
            'mobile_usage_while_driving': {'confidence_threshold': 0.75},

            # Parking violations
            'no_parking_violation': {'confidence_threshold': 0.80},
            'double_parking': {'confidence_threshold': 0.85},
            'footpath_parking': {'confidence_threshold': 0.80},

            # Movement violations
            'wrong_side_driving': {'confidence_threshold': 0.85},
            'reverse_driving': {'confidence_threshold': 0.80},
            'u_turn_violation': {'confidence_threshold': 0.75},

            # Vehicle violations
            'overloading': {'confidence_threshold': 0.75},
            'triple_riding': {'confidence_threshold': 0.80},
            'goods_vehicle_violation': {'confidence_threshold': 0.70},

            # Emergency violations
            'ambulance_not_giving_way': {'confidence_threshold': 0.85},
            'emergency_lane_violation': {'confidence_threshold': 0.80},

            # Indian specific
            'auto_rickshaw_violation': {'confidence_threshold': 0.75},
            'cycle_on_highway': {'confidence_threshold': 0.80},
            'animal_on_road': {'confidence_threshold': 0.70},
            'jaywalking': {'confidence_threshold': 0.75},
            'encroachment': {'confidence_threshold': 0.70}
        }

        self.zones = {
            'Zone A': {'x1': 0, 'y1': 0, 'x2': 640, 'y2': 360},
            'Zone B': {'x1': 640, 'y1': 0, 'x2': 1280, 'y2': 360},
            'Zone C': {'x1': 0, 'y1': 360, 'x2': 640, 'y2': 720},
            'Zone D': {'x1': 640, 'y1': 360, 'x2': 1280, 'y2': 720}
        }

        self.tracking_data = defaultdict(lambda: deque(maxlen=30))
        self.violation_history = []
        self.frame_count = 0
        self.fps = 30

    def detect_vehicles_and_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles, pedestrians, and other objects using YOLO or similar model
        This would typically use a pre-trained model like YOLOv8
        """
        # Placeholder for actual object detection
        # In practice, you would use models like:
        # - YOLOv8 for general object detection
        # - Custom trained models for Indian traffic scenarios

        detections = []

        # Simulate detection results for demonstration
        # Replace with actual model inference
        height, width = frame.shape[:2]

        # Example detections (replace with actual model output)
        sample_detections = [
            {
                'class': 'car',
                'bbox': [100, 200, 200, 300],
                'confidence': 0.92,
                'track_id': 1
            },
            {
                'class': 'motorcycle',
                'bbox': [300, 250, 350, 320],
                'confidence': 0.88,
                'track_id': 2
            },
            {
                'class': 'person',
                'bbox': [450, 300, 480, 400],
                'confidence': 0.85,
                'track_id': 3
            }
        ]

        return sample_detections

    def detect_traffic_lights(self, frame: np.ndarray) -> Dict:
        """
        Detect traffic light status using color detection and ML models
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for traffic lights
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])

        yellow_lower = np.array([20, 50, 50])
        yellow_upper = np.array([30, 255, 255])

        # Create masks
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        # Determine signal state
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)

        if red_pixels > green_pixels and red_pixels > yellow_pixels:
            return {'status': 'red', 'confidence': 0.8}
        elif green_pixels > red_pixels and green_pixels > yellow_pixels:
            return {'status': 'green', 'confidence': 0.8}
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            return {'status': 'yellow', 'confidence': 0.8}
        else:
            return {'status': 'unknown', 'confidence': 0.3}

    def detect_lane_markings(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detect lane markings using edge detection and Hough transforms
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Define region of interest (lower half of frame typically)
        height, width = edges.shape
        roi_vertices = np.array([
            [(0, height), (width//2, height//2), (width, height)]
        ], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255) # type: ignore
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi/180, threshold=50,
            minLineLength=100, maxLineGap=50
        )

        return lines if lines is not None else []

    def calculate_speed(self, track_id: int, current_pos: Tuple[int, int],
                        timestamp: datetime) -> float:
        """
        Calculate vehicle speed based on position tracking
        """
        if track_id not in self.tracking_data:
            self.tracking_data[track_id].append((current_pos, timestamp))
            return 0.0

        prev_data = self.tracking_data[track_id]
        if len(prev_data) < 2:
            prev_data.append((current_pos, timestamp))
            return 0.0

        # Calculate speed using last two positions
        prev_pos, prev_time = prev_data[-1]
        current_data = (current_pos, timestamp)
        prev_data.append(current_data)

        # Calculate distance (assuming pixel to meter conversion)
        pixel_to_meter = 0.05  # Calibration factor
        distance = math.sqrt(
            (current_pos[0] - prev_pos[0])**2 +
            (current_pos[1] - prev_pos[1])**2
        ) * pixel_to_meter

        # Calculate time difference
        time_diff = (timestamp - prev_time).total_seconds()

        if time_diff > 0:
            speed = distance / time_diff  # m/s
            return speed * 3.6  # Convert to km/h

        return 0.0

    def detect_helmet_seatbelt(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Detect helmet and seatbelt usage using specialized models
        """
        violations = []

        for detection in detections:
            if detection['class'] in ['motorcycle', 'scooter']:
                # Extract motorcycle region
                x1, y1, x2, y2 = detection['bbox']
                vehicle_roi = frame[y1:y2, x1:x2]

                # Helmet detection logic (placeholder)
                # In practice, use trained models for helmet detection
                helmet_detected = self._detect_helmet_in_roi(vehicle_roi)

                if not helmet_detected and detection['confidence'] > 0.8:
                    violations.append({
                        'type': 'no_helmet',
                        'track_id': detection['track_id'],
                        'confidence': 0.85
                    })

            elif detection['class'] in ['car', 'suv', 'sedan']:
                # Seatbelt detection for cars
                x1, y1, x2, y2 = detection['bbox']
                vehicle_roi = frame[y1:y2, x1:x2]

                seatbelt_detected = self._detect_seatbelt_in_roi(vehicle_roi)

                if not seatbelt_detected and detection['confidence'] > 0.8:
                    violations.append({
                        'type': 'no_seatbelt',
                        'track_id': detection['track_id'],
                        'confidence': 0.80
                    })

        return violations

    def _detect_helmet_in_roi(self, roi: np.ndarray) -> bool:
        """
        Placeholder for helmet detection in ROI
        """
        # Implement actual helmet detection model here
        return np.random.random() > 0.3  # Placeholder

    def _detect_seatbelt_in_roi(self, roi: np.ndarray) -> bool:
        """
        Placeholder for seatbelt detection in ROI
        """
        # Implement actual seatbelt detection model here
        return np.random.random() > 0.4  # Placeholder

    def detect_mobile_usage(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Detect mobile phone usage while driving
        """
        violations = []

        for detection in detections:
            if detection['class'] in ['car', 'motorcycle']:
                x1, y1, x2, y2 = detection['bbox']
                vehicle_roi = frame[y1:y2, x1:x2]

                # Mobile detection logic (placeholder)
                mobile_detected = self._detect_mobile_in_roi(vehicle_roi)

                if mobile_detected:
                    violations.append({
                        'type': 'mobile_usage_while_driving',
                        'track_id': detection['track_id'],
                        'confidence': 0.75
                    })

        return violations

    def _detect_mobile_in_roi(self, roi: np.ndarray) -> bool:
        """
        Placeholder for mobile phone detection
        """
        return np.random.random() > 0.8  # Placeholder

    def detect_lane_violations(self, detections: List[Dict], lanes: List[np.ndarray]) -> List[Dict]:
        """
        Detect lane cutting and wrong lane violations
        """
        violations = []

        for detection in detections:
            track_id = detection['track_id']
            current_pos = self._get_center_point(detection['bbox'])

            # Check for sudden lane changes (lane cutting)
            if track_id in self.tracking_data:
                prev_positions = list(self.tracking_data[track_id])
                if len(prev_positions) >= 5:
                    # Analyze movement pattern
                    lateral_movement = self._calculate_lateral_movement(
                        prev_positions)

                    if lateral_movement > 50:  # Sudden lateral movement
                        violations.append({
                            'type': 'lane_cutting',
                            'track_id': track_id,
                            'confidence': 0.85
                        })

        return violations

    def _get_center_point(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Get center point of bounding box
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_lateral_movement(self, positions: List[Tuple]) -> float:
        """
        Calculate lateral movement from position history
        """
        if len(positions) < 2:
            return 0.0

        x_positions = [pos[0][0] for pos in positions[-5:]]
        return max(x_positions) - min(x_positions)

    def detect_signal_violations(self, detections: List[Dict],
                                 traffic_light: Dict, lanes: List[np.ndarray]) -> List[Dict]:
        """
        Detect red light jumping and other signal violations
        """
        violations = []

        if traffic_light['status'] == 'red' and traffic_light['confidence'] > 0.7:
            for detection in detections:
                track_id = detection['track_id']
                current_pos = self._get_center_point(detection['bbox'])

                # Check if vehicle crossed stop line during red signal
                if self._crossed_stop_line(current_pos, lanes):
                    violations.append({
                        'type': 'red_light_jumping',
                        'track_id': track_id,
                        'confidence': 0.90
                    })

        return violations

    def _crossed_stop_line(self, position: Tuple[int, int], lanes: List[np.ndarray]) -> bool:
        """
        Check if vehicle crossed the stop line
        """
        # Placeholder logic for stop line detection
        return False

    def get_zone_from_position(self, position: Tuple[int, int]) -> str:
        """
        Determine which zone a position belongs to
        """
        x, y = position

        for zone_name, zone_coords in self.zones.items():
            if (zone_coords['x1'] <= x <= zone_coords['x2'] and
                    zone_coords['y1'] <= y <= zone_coords['y2']):
                return zone_name

        return 'Zone A'  # Default zone

    def process_frame(self, frame: np.ndarray, timestamp: datetime) -> List[Dict]:
        """
        Process a single frame and detect violations
        """
        violations = []

        # Detect vehicles and objects
        detections = self.detect_vehicles_and_objects(frame)

        # Detect traffic lights
        traffic_light = self.detect_traffic_lights(frame)

        # Detect lane markings
        lanes = self.detect_lane_markings(frame)

        # Process each detection for violations
        for detection in detections:
            track_id = detection['track_id']
            current_pos = self._get_center_point(detection['bbox'])
            zone = self.get_zone_from_position(current_pos)

            # Calculate speed
            speed = self.calculate_speed(track_id, current_pos, timestamp)

            # Check for overspeeding
            if speed > self.violation_types['overspeeding']['speed_limit']:
                violations.append({
                    'timestamp': timestamp.isoformat(),
                    'event_type': 'overspeeding',
                    'confidence': 0.90,
                    'location': zone,
                    'details': {'speed': speed, 'track_id': track_id}
                })

        # Detect helmet/seatbelt violations
        safety_violations = self.detect_helmet_seatbelt(frame, detections)
        for violation in safety_violations:
            violations.append({
                'timestamp': timestamp.isoformat(),
                'event_type': violation['type'],
                'confidence': violation['confidence'],
                'location': self.get_zone_from_position(
                    self._get_center_point(
                        next(d['bbox'] for d in detections
                             if d['track_id'] == violation['track_id'])
                    )
                )
            })

        # Detect mobile usage violations
        mobile_violations = self.detect_mobile_usage(frame, detections)
        for violation in mobile_violations:
            violations.append({
                'timestamp': timestamp.isoformat(),
                'event_type': violation['type'],
                'confidence': violation['confidence'],
                'location': self.get_zone_from_position(
                    self._get_center_point(
                        next(d['bbox'] for d in detections
                             if d['track_id'] == violation['track_id'])
                    )
                )
            })

        # Detect lane violations
        lane_violations = self.detect_lane_violations(detections, lanes)
        for violation in lane_violations:
            violations.append({
                'timestamp': timestamp.isoformat(),
                'event_type': violation['type'],
                'confidence': violation['confidence'],
                'location': self.get_zone_from_position(
                    self._get_center_point(
                        next(d['bbox'] for d in detections
                             if d['track_id'] == violation['track_id'])
                    )
                )
            })

        # Detect signal violations
        signal_violations = self.detect_signal_violations(
            detections, traffic_light, lanes)
        for violation in signal_violations:
            violations.append({
                'timestamp': timestamp.isoformat(),
                'event_type': violation['type'],
                'confidence': violation['confidence'],
                'location': self.get_zone_from_position(
                    self._get_center_point(
                        next(d['bbox'] for d in detections
                             if d['track_id'] == violation['track_id'])
                    )
                )
            })

        self.frame_count += 1
        return violations


    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process video and detect violations every 0.5 seconds.
        """
        cap = cv2.VideoCapture(video_path)
        all_violations = []

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else 30
        # Process one frame every 0.5 seconds
        frame_interval = max(1, int(self.fps // 6))

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                # Calculate timestamp
                timestamp = datetime.now() + timedelta(seconds=frame_number / self.fps)

                # Process this frame
                violations = self.process_frame(frame, timestamp)
                all_violations.extend(violations)

            frame_number += 1

        cap.release()
        return all_violations



    def save_violations_to_json(self, violations: List[Dict], output_path: str):
        """
        Save detected violations to JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(violations, f, indent=2, default=str)

        print(f"Violations saved to {output_path}")

# Usage example


def VoilationDetection(videoPath: str):
    # Initialize detector
    detector = IndianTrafficViolationDetector()

    # Process video
    try:
        violations = detector.process_video(videoPath)

        print(f"Detected {len(violations)} violations:")
        for violation in violations[:10]:
            print(f"- {violation['event_type']} at {violation['timestamp']} "
                  f"(confidence: {violation['confidence']:.2f}, location: {violation['location']})")

        # Save to JSON
        output_file = "detected_violations.json"
        detector.save_violations_to_json(violations, output_file)

        return output_file  # ✅ Return the file path

    except Exception as e:
        print(f"Error processing video: {e}")
        return None
