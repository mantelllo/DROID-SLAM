import numpy as np
from pyvista import Sphere
from scipy.spatial.transform import Rotation as R
import time
import pyvista as pv
from vispy import scene, app
import torch
from vispy.scene.cameras import ArcballCamera
from vispy.scene import TurntableCamera
from vispy.visuals.transforms import STTransform


class VisPyViewer:
    def __init__(self):
        # 1) Create a non-blocking 3D canvas
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        self.view   = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'    # orbit/zoom with mouse

        # in your __init__():
        cam = TurntableCamera(
            fov=60,  # field of view in degrees
            center=[-0.309697 ,  0.0960165,  1.2706246],  # pivot point
            up='-y',
        )
        # cam.scale_factor = 0.5  # slows the drag so small mouse moves spin more
        self.view.camera = cam

        # draw XYZ axes so you have a frame of reference
        scene.visuals.XYZAxis(parent=self.view.scene)

        # 2) Add an initially empty scatter plot with depth testing ON
        self.scatter = scene.visuals.Markers(parent=self.view.scene, scaling=False)
        self.scatter.set_gl_state('translucent', depth_test=True)
        self.scatter.set_data(np.empty((0,3)), face_color=(1,1,1,1), size=0.01)

        # ensure the first frame actually gets drawn
        app.process_events()

    def update(self, points: np.ndarray, poses):
        """
        points: (N×6) array of xyzRGB
        poses:  iterable of (7,) arrays [x, y, z, qx, qy, qz, qw]
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        # --- draw point cloud as before ---
        xyz, c = points[:, :3], points[:, 3:6]
        # xyz[:, 0] *= -1.0

        if c.max() > 1.0:
            c = c / 255.0
        c = c[:, [2, 1, 0]]
        alpha = np.ones((len(c), 1))
        rgba = np.hstack((c, alpha))
        self.scatter.set_data(xyz, face_color=rgba, edge_color=rgba, size=0.01)

        # --- lazy‐init camera visuals ---
        if not hasattr(self, 'cam_scatter'):
            from vispy import scene
            self.cam_scatter = scene.visuals.Markers(parent=self.canvas.scene)
            self.cam_dirs = scene.visuals.Line(
                parent=self.canvas.scene, connect='segments', width=2
            )
        #
        # # --- draw camera poses from 7-DOF ---
        # poses = list(poses)
        # if poses:
        #     # 1) positions (red dots)
        #     cam_pos = np.array([p[:3] for p in poses])
        #     # cam_pos[:, 1] *= -1.0
        #     self.cam_scatter.set_data(
        #         cam_pos,
        #         face_color=(1, 0, 0, 1),
        #         edge_color=(1, 0, 0, 1),
        #         size=0.03
        #     )
        #
        #     # 2) forward-direction segments (green lines)
        #     segs = []
        #     for i, p in enumerate(poses):
        #         x, y, z, qx, qy, qz, qw = p
        #         # normalize quaternion
        #         norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        #         qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        #
        #         # rotate camera’s local +Z = forward vector
        #         fwd = np.array([
        #             2 * (qx * qz + qy * qw),
        #             2 * (qy * qz - qx * qw),
        #             1 - 2 * (qx * qx + qy * qy)
        #         ])
        #         fwd[1] *= -1.0  # match Y-flip
        #
        #         start = cam_pos[i]
        #         end = start + fwd * 0.1  # adjust length as needed
        #         segs.extend([start, end])
        #
        #     pts = np.vstack(segs)
        #     self.cam_dirs.set_data(pts, connect='segments', color=(0, 1, 0, 1))

        # redraw
        self.canvas.update()
        app.process_events()


def start_viewer():
    return VisPyViewer()
