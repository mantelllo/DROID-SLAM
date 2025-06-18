import numpy as np
import open3d as o3d

# 1. Load and sanity-check
data = np.load("/home/john/git/aerial_gym_ws/src/aerial_gym_simulator/trees.npy")           # expect shape (N,6)
if data.ndim != 2 or data.shape[1] != 6:
    raise ValueError(f"Expected shape (N,6), got {data.shape}")

# 2. Split, convert dtype, enforce C-contiguity
pts = np.ascontiguousarray(data[:, :3], dtype=np.float64)           # (N,3) doubles
bgr = np.ascontiguousarray(data[:, 3:6], dtype=np.float64)          # (N,3) doubles
rgb = bgr[:, ::-1]                               # swap BGR→RGB


pts[:,1] = -pts[:,1]


# 3. Build PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)                        # safe now
pcd.colors = o3d.utility.Vector3dVector(rgb)                        # must match N points

# # 4. (Optional) Outlier removal
# pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
# pcd = pcd.select_by_index(ind)

# 5. Visualize with orthographic projection
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Trees (ortho)", width=800, height=600)
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.point_size = 1.0

vc = vis.get_view_control()
# vc.set_projection_type(o3d.visualization.ViewControl.PROJECTION_ORTHO)

vis.run()
vis.destroy_window()
