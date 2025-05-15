import bpy
import os
import numpy as np
import sys
from scipy.spatial import procrustes

def convert_blendshapes_to_vertices(input_path, output_path, obj_name="face", shape_key_names=None, 
                                    apply_rigid_align=False, ref_vertices_path=None):
    """
    将存储 blendshape 参数的 .npy 文件转换为 3D 顶点坐标，并保存为 .npy 文件。

    参数：
      input_path: str，blendshape 参数文件路径（.npy 文件，形状为 (N_frames, 52)）
      output_path: str，转换后 3D 顶点数据保存路径
      obj_name: str，Blender 中的对象名称，该对象应含有对应的 Shape Keys
      shape_key_names: list of str，52 个 Shape Keys 的名称，顺序必须与 blendshape 参数顺序一致
      apply_rigid_align: bool，是否对转换的顶点进行刚性对齐（仅针对预测数据）
      ref_vertices_path: str，当 apply_rigid_align 为 True 时，指定对应的 ground truth 3D 顶点文件，
                         文件应与当前处理的预测文件同名且存放在另一个文件夹中，
                         数组形状为 (N_frames, num_vertices, 3)
    """
    if shape_key_names is None:
        shape_key_names = [
            "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
            "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
            "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
            "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
            "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
            "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
            "tongueOut"
        ]

    # 加载 blendshape 参数数据
    blendshape_coeffs = np.load(input_path)
    num_frames, num_keys = blendshape_coeffs.shape
    print("Total frames:", num_frames)

    # 如果需要刚性对齐，加载对应的 ground truth 3D 顶点数据
    if apply_rigid_align:
        if ref_vertices_path is None:
            raise ValueError("当 apply_rigid_align 为 True 时，必须指定 ref_vertices_path 参数。")
        ref_vertices_all = np.load(ref_vertices_path)
        if ref_vertices_all.shape[0] != num_frames:
            raise ValueError("ground truth 数据的帧数与输入的 blendshape 参数帧数不匹配。")

    # 获取 Blender 中的对象
    if obj_name not in bpy.data.objects:
        raise ValueError(f"Object named '{obj_name}' not found in the scene.")
    face_obj = bpy.data.objects[obj_name]

    all_vertices = []

    for frame_idx in range(num_frames):
        curr_coeffs = blendshape_coeffs[frame_idx]
        # 设置每个 shape key 的值
        for j, key_name in enumerate(shape_key_names):
            if key_name in face_obj.data.shape_keys.key_blocks:
                face_obj.data.shape_keys.key_blocks[key_name].value = curr_coeffs[j]
            else:
                print(f"Warning: Shape key '{key_name}' not found.")
        bpy.context.view_layer.update()
        
        # 获取经过变形后的网格（经过依赖图计算后的结果）
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = face_obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        vertices = np.array([v.co[:] for v in mesh.vertices])
        
        # 如果是预测数据，进行刚性对齐
        if apply_rigid_align:
            gt_vertices = ref_vertices_all[frame_idx]
            _, aligned_vertices, _ = procrustes(gt_vertices, vertices)
            vertices = aligned_vertices
        
        all_vertices.append(vertices)
        eval_obj.to_mesh_clear()
        
        if frame_idx % 10 == 0:
            print("Processed frame:", frame_idx)

    all_vertices = np.stack(all_vertices, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_vertices)
    print("Saved 3D vertices to:", output_path)


if __name__ == "__main__":
    # 利用 "--" 分隔 Blender 内置参数和传递给脚本的参数
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx+1:]
    except ValueError:
        raise ValueError("必须使用 '--' 分隔 Blender 内置参数和传递给脚本的参数。")
    
    # 解析参数：要求有两个位置参数（input, output），
    # 同时支持可选标志 --ref_dir 后跟一个路径
    positional = []
    ref_dir = None
    i = 0
    while i < len(args):
        # 如果遇到标志 --ref_dir，下一项则为对应的目录
        if args[i] == "--ref_dir":
            if i + 1 < len(args):
                ref_dir = args[i+1]
                i += 2
            else:
                raise ValueError("参数 --ref_dir 后面缺少对应的目录路径")
        else:
            positional.append(args[i])
            i += 1
    
    if len(positional) != 2:
        raise ValueError("Usage: blender --background --python blendshape_to_3d.py -- <input> <output> [--ref_dir <ref_dir>]")
    
    input_path = positional[0]
    output_path = positional[1]
    
    # 判断是文件还是目录
    if os.path.isdir(input_path):
        # 如果输入的是目录，则输出也应为目录
        input_items = [f for f in os.listdir(input_path) if f.endswith('.npy')]
        for f in input_items:
            in_file = os.path.join(input_path, f)
            out_file = os.path.join(output_path, f)
            ref_file = None
            if ref_dir:
                ref_file = os.path.join(ref_dir, f)
                if not os.path.exists(ref_file):
                    raise ValueError(f"没有找到与 {f} 对应的 ground truth 文件：{ref_file}")
            print("Processing file:", in_file)
            convert_blendshapes_to_vertices(in_file, out_file, obj_name="face", 
                                              apply_rigid_align=(ref_file is not None),
                                              ref_vertices_path=ref_file)
    else:
        # 单个文件转换
        ref_file = None
        if ref_dir:
            ref_file = os.path.join(ref_dir, os.path.basename(input_path))
            if not os.path.exists(ref_file):
                raise ValueError(f"没有找到与 {input_path} 对应的 ground truth 文件：{ref_file}")
        convert_blendshapes_to_vertices(input_path, output_path, obj_name="face", 
                                        apply_rigid_align=(ref_file is not None),
                                        ref_vertices_path=ref_file)

