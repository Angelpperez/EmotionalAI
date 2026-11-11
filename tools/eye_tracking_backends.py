"""Interchangeable ocular-localization backends for the live experiment."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def is_placeholder_frame(frame: np.ndarray) -> bool:
    """Identify near-black virtual-camera status frames with sparse icons."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_fraction = float(np.mean(gray < 10))
    return float(np.mean(gray)) < 5.0 and dark_fraction > 0.95


@dataclass
class EyeTrackingResult:
    angles: list[float]
    display: np.ndarray
    face_detected: bool = False
    eye_count: int = 0
    pupil_count: int = 0
    face_bbox: tuple[int, int, int, int] | None = None
    eye_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    eye_openness: list[float] = field(default_factory=list)
    normalized_iris_offsets: list[tuple[float, float]] = field(
        default_factory=list
    )


class HaarContourEyeTracker:
    """Preserve the original Haar plus dark-contour detector."""

    name = "haar_contour"

    def __init__(self) -> None:
        from live_vreed_eye_state_udp import (
            detect_pupil_in_eye,
            eye_cascade,
            face_cascade,
        )

        self.detect_pupil_in_eye = detect_pupil_in_eye
        self.eye_cascade = eye_cascade
        self.face_cascade = face_cascade

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "opencv_version": cv2.__version__,
            "method": "Haar face/eye cascades plus thresholded contour ellipse",
        }

    def process(self, frame: np.ndarray) -> EyeTrackingResult:
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )
        result = EyeTrackingResult(angles=[], display=display)
        if not len(faces):
            return result

        x, y, width, height = max(
            faces,
            key=lambda item: int(item[2]) * int(item[3]),
        )
        result.face_detected = True
        result.face_bbox = (int(x), int(y), int(width), int(height))
        cv2.rectangle(
            display,
            (x, y),
            (x + width, y + height),
            (255, 160, 0),
            2,
        )
        roi_gray = gray[y : y + height, x : x + width]
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 20),
        )
        candidates = []
        for ex, ey, ew, eh in eyes:
            if ey + eh / 2 > 0.6 * height:
                continue
            ratio = ew / float(eh)
            if 0.7 <= ratio <= 3.0:
                candidates.append((ex, ey, ew, eh))
        candidates = sorted(
            sorted(candidates, key=lambda item: -item[2])[:2],
            key=lambda item: item[0],
        )

        result.eye_count = len(candidates)
        for ex, ey, ew, eh in candidates:
            box = (x + ex, y + ey, int(ew), int(eh))
            result.eye_bboxes.append(box)
            eye_gray = roi_gray[ey : ey + eh, ex : ex + ew]
            center, radius = self.detect_pupil_in_eye(eye_gray)
            color = (0, 180, 255)
            if (
                center is not None
                and radius is not None
                and 3 <= radius <= min(ew, eh) // 2
            ):
                pupil_x = x + ex + center[0]
                pupil_y = y + ey + center[1]
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                vector_x = pupil_x - eye_center_x
                vector_y = pupil_y - eye_center_y
                result.angles.append(
                    float(np.degrees(np.arctan2(-vector_y, vector_x)))
                )
                result.pupil_count += 1
                color = (0, 255, 0)
                cv2.circle(
                    display,
                    (pupil_x, pupil_y),
                    max(3, int(radius)),
                    color,
                    2,
                )
            cv2.rectangle(
                display,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                color,
                2,
            )
        return result

    def close(self) -> None:
        return None


class MediaPipeIrisTracker:
    """Track refined face and iris landmarks with MediaPipe Face Mesh."""

    name = "mediapipe_face_mesh_iris"

    # Indices follow MediaPipe's 478-landmark refined Face Mesh topology.
    EYES = (
        {
            "corners": (33, 133),
            "upper": 159,
            "lower": 145,
            "iris_center": 468,
            "iris_ring": (469, 470, 471, 472),
            "contour": (33, 160, 158, 133, 153, 144),
        },
        {
            "corners": (362, 263),
            "upper": 386,
            "lower": 374,
            "iris_center": 473,
            "iris_ring": (474, 475, 476, 477),
            "contour": (362, 385, 387, 263, 373, 380),
        },
    )

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_eye_openness: float = 0.04,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Use "
                "D:\\emotion_eye_env312\\Scripts\\python.exe."
            ) from exc

        self.mp = mp
        self.min_eye_openness = float(min_eye_openness)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "mediapipe_version": self.mp.__version__,
            "opencv_version": cv2.__version__,
            "landmark_count": 478,
            "refine_landmarks": True,
            "min_eye_openness": self.min_eye_openness,
            "signal": (
                "polar angle of the iris-center displacement in each "
                "eye-local coordinate frame"
            ),
        }

    @staticmethod
    def _pixel_points(
        landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:
        return np.asarray(
            [
                (landmark.x * width, landmark.y * height)
                for landmark in landmarks
            ],
            dtype=np.float64,
        )

    def process(self, frame: np.ndarray) -> EyeTrackingResult:
        display = frame.copy()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        detection = self.face_mesh.process(rgb)
        result = EyeTrackingResult(angles=[], display=display)
        if not detection.multi_face_landmarks:
            return result

        points = self._pixel_points(
            detection.multi_face_landmarks[0].landmark,
            width,
            height,
        )
        if points.shape[0] < 478:
            return result

        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int)
        x0 = int(np.clip(x0, 0, width - 1))
        y0 = int(np.clip(y0, 0, height - 1))
        x1 = int(np.clip(x1, x0 + 1, width))
        y1 = int(np.clip(y1, y0 + 1, height))
        result.face_detected = True
        result.face_bbox = (x0, y0, x1 - x0, y1 - y0)
        cv2.rectangle(display, (x0, y0), (x1, y1), (255, 160, 0), 2)

        for eye in self.EYES:
            corner_points = points[list(eye["corners"])]
            upper = points[int(eye["upper"])]
            lower = points[int(eye["lower"])]
            eye_width = float(np.linalg.norm(corner_points[1] - corner_points[0]))
            eye_height = float(np.linalg.norm(lower - upper))
            if eye_width <= 1.0:
                continue

            openness = eye_height / eye_width
            contour = points[list(eye["contour"])]
            bx0, by0 = np.floor(contour.min(axis=0)).astype(int)
            bx1, by1 = np.ceil(contour.max(axis=0)).astype(int)
            padding = max(2, int(round(eye_width * 0.08)))
            bx0 = int(np.clip(bx0 - padding, 0, width - 1))
            by0 = int(np.clip(by0 - padding, 0, height - 1))
            bx1 = int(np.clip(bx1 + padding, bx0 + 1, width))
            by1 = int(np.clip(by1 + padding, by0 + 1, height))
            result.eye_bboxes.append((bx0, by0, bx1 - bx0, by1 - by0))
            result.eye_openness.append(openness)
            result.eye_count += 1

            contour_int = np.rint(contour).astype(np.int32)
            cv2.polylines(
                display,
                [contour_int.reshape((-1, 1, 2))],
                True,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            if openness < self.min_eye_openness:
                cv2.rectangle(
                    display,
                    (bx0, by0),
                    (bx1, by1),
                    (0, 140, 255),
                    1,
                )
                continue

            iris_center = points[int(eye["iris_center"])]
            iris_ring = points[list(eye["iris_ring"])]
            eye_center = np.mean(
                np.vstack((corner_points, upper, lower)),
                axis=0,
            )

            horizontal = corner_points[1] - corner_points[0]
            if horizontal[0] < 0:
                horizontal = -horizontal
            horizontal /= np.linalg.norm(horizontal)
            vertical = np.asarray(
                (-horizontal[1], horizontal[0]),
                dtype=np.float64,
            )
            offset = iris_center - eye_center
            local_x = float(np.dot(offset, horizontal))
            local_y = float(np.dot(offset, vertical))
            result.normalized_iris_offsets.append(
                (local_x / eye_width, local_y / eye_width)
            )
            result.angles.append(
                float(np.degrees(np.arctan2(-local_y, local_x)))
            )
            result.pupil_count += 1

            radius = max(
                2,
                int(
                    round(
                        np.mean(
                            np.linalg.norm(
                                iris_ring - iris_center,
                                axis=1,
                            )
                        )
                    )
                ),
            )
            center_int = tuple(np.rint(iris_center).astype(int))
            eye_center_int = tuple(np.rint(eye_center).astype(int))
            cv2.circle(display, center_int, radius, (0, 255, 0), 2)
            cv2.circle(display, center_int, 2, (0, 255, 255), -1)
            cv2.line(
                display,
                eye_center_int,
                center_int,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return result

    def close(self) -> None:
        self.face_mesh.close()


def create_eye_tracker(name: str):
    normalized = name.strip().lower()
    if normalized in {"haar", "haar_contour"}:
        return HaarContourEyeTracker()
    if normalized in {"mediapipe", "mediapipe_face_mesh_iris"}:
        return MediaPipeIrisTracker()
    raise ValueError(f"Unsupported eye detector: {name}")
