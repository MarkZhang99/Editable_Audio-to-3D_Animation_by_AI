import numpy as np
import trimesh
import pyrender

def visualize_frame(vertices_file,frame_index):
    """
    加载3D顶点数据和面数据，并可视化指定帧的网格。

    参数：
      vertices_file: str，包含3D顶点数据的.npy文件路径，形状应为 (F, N, 3)
      faces_file: str，包含网格面数据的.npy文件路径，形状应为 (num_faces, 3)
      frame_index: int，要可视化的帧索引

    返回：
      无返回值，通过 pyrender 打开一个可视化窗口显示网格
    """
    faces_file = "./result/3d_result/oldtest/faces_tri.npy"
    # 加载3D顶点数据
    vertices_all = np.load(vertices_file)
    print("Vertices shape:", vertices_all.shape)
    
    # 检查 frame_index 是否在范围内
    if frame_index < 0 or frame_index >= vertices_all.shape[0]:
        raise ValueError(f"frame_index {frame_index} 超出有效范围 [0, {vertices_all.shape[0]-1}]")
    
    # 取出指定帧的顶点数据
    vertices = vertices_all[frame_index]  # 形状 (N, 3)
    
    # 加载面数据（faces）
    faces = np.load(faces_file)
    
    # 创建 trimesh 对象，并转换为 pyrender Mesh
    tri_mesh = trimesh.Trimesh(vertices, faces)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    
    # 创建场景并添加网格
    scene = pyrender.Scene()
    scene.add(mesh)
    
    # 启动 pyrender 可视化窗口
    pyrender.Viewer(scene, use_raymond_lighting=True)

# 示例调用
if __name__ == "__main__":
    vertices_file = "./result/3d_result/disgust.npy"
    
    frame_index = 50  # 你可以修改为你想查看的帧数
    visualize_frame(vertices_file, faces_file, frame_index)
