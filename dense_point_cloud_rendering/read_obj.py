
import taichi as ti
import numpy as np

def load_obj_vertices(obj_path: str, max_verts: int, scale: float = 1.0):
    """
    返回:
        vert_field : ti.field   (max_verts, 3)
        vert_count : int
        tri_field  : ti.field   (num_tri, 3)  类型 i32
    """
    # ================= 第一遍：读取顶点 =================
    verts_arr = np.zeros((max_verts, 3), dtype=np.float32)
    vert_count = 0
    # 同时统计三角形数量（为预分配做准备）
    tri_count = 0
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts_arr[vert_count, 0] = float(parts[1])
                verts_arr[vert_count, 1] = float(parts[2])
                verts_arr[vert_count, 2] = float(parts[3])
                vert_count += 1
                if vert_count == max_verts:
                    break
            # 顺便统计三角形数量
            elif line.startswith('f '):
                parts = line.split()[1:]   # 顶点字段
                if len(parts) == 3:        # 只统计三角形
                    tri_count += 1

    # 应用缩放
    if scale != 1.0:
        verts_arr[:vert_count] *= scale

    # 创建顶点场
    vert_field = ti.field(dtype=ti.f32, shape=(max_verts, 3))
    vert_field.from_numpy(verts_arr)
    print(f'{obj_path} read success: {vert_count} vertices')

    # ================= 第二遍：读取三角形面 =================
    # 预分配 NumPy 数组，直接按索引填入，无 append
    tri_arr = np.zeros((tri_count, 3), dtype=np.int32)
    tri_idx = 0
    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('f '):
                continue
            parts = line.split()[1:]
            if len(parts) != 3:
                continue

            # 解析三个顶点索引，转为 0 基，处理负索引
            v0 = int(parts[0].split('/')[0])
            v1 = int(parts[1].split('/')[0])
            v2 = int(parts[2].split('/')[0])

            # 负索引转换
            if v0 < 0: v0 = vert_count + v0 + 1
            if v1 < 0: v1 = vert_count + v1 + 1
            if v2 < 0: v2 = vert_count + v2 + 1

            tri_arr[tri_idx, 0] = v0 - 1
            tri_arr[tri_idx, 1] = v1 - 1
            tri_arr[tri_idx, 2] = v2 - 1
            tri_idx += 1

    # 从 NumPy 直接创建三角形场
    tri_field = ti.field(dtype=ti.i32, shape=(tri_count, 3))
    tri_field.from_numpy(tri_arr)
    print(f'Extracted {tri_count} triangles into Taichi field')

    return vert_field, tri_field,vert_count
import taichi as ti
import numpy as np
import taichi as ti
import numpy as np

@ti.func
def rand2() -> ti.f32:
    """返回一个 (0,1) 之间的随机浮点数"""
    return ti.random(ti.f32)

@ti.kernel
def fill_points_random_kernel(
    verts: ti.types.ndarray(ndim=2, dtype=ti.f32),
    tris: ti.types.ndarray(ndim=2, dtype=ti.i32),
    point_tri_ids: ti.types.ndarray(ndim=1, dtype=ti.i32),
    point_seeds: ti.types.ndarray(ndim=1, dtype=ti.f32),
    output: ti.types.ndarray(ndim=2, dtype=ti.f32)
):
    """使用随机采样的kernel，更快更均匀"""
    for i in range(output.shape[0]):
        tri_id = point_tri_ids[i]
        seed = point_seeds[i]

        v0 = tris[tri_id, 0]
        v1 = tris[tri_id, 1]
        v2 = tris[tri_id, 2]
        A = ti.Vector([verts[v0, 0], verts[v0, 1], verts[v0, 2]])
        B = ti.Vector([verts[v1, 0], verts[v1, 1], verts[v1, 2]])
        C = ti.Vector([verts[v2, 0], verts[v2, 1], verts[v2, 2]])

        # 使用确定性伪随机（基于seed）
        r1 = ti.sqrt(ti.random(ti.f32) * 0.5 + seed * 0.5)
        r2 = ti.random(ti.f32) * 0.5 + seed * 0.5
        u = 1.0 - r1
        v = r1 * (1.0 - r2)
        w = r1 * r2

        P = u * A + v * B + w * C

        output[i, 0] = P[0]
        output[i, 1] = P[1]
        output[i, 2] = P[2]

def fill_face(vert_field: ti.field, tri_field: ti.field, spacing: float) -> ti.field:
    """
    基于间距的随机采样版本（推荐使用）

    参数:
        spacing: 点间距（越小越密集）
                 例如 spacing=0.01 会生成很密的点
                 spacing=0.1 会生成稀疏的点
    """
    verts_np = vert_field.to_numpy()
    tris_np = tri_field.to_numpy()
    M = tris_np.shape[0]

    # 计算三角形面积
    v0 = verts_np[tris_np[:, 0]]
    v1 = verts_np[tris_np[:, 1]]
    v2 = verts_np[tris_np[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    # 基于间距计算每个三角形的点数
    # 正三角形的面积公式：A = (√3/4) * a²，其中a是边长
    # 在间距为spacing的规则网格中，每个点占据的面积约为 spacing² * √3/2
    point_area = spacing * spacing * np.sqrt(3) / 2
    points_per_tri = np.maximum(1, (areas / point_area).astype(np.int32))

    total_points = points_per_tri.sum()

    print(f"采样间距: {spacing}")
    print(f"总三角形数: {M}")
    print(f"总采样点数: {total_points}")
    print(f"平均点密度: {total_points / areas.sum():.1f} 点/平方单位")

    # 构造映射
    point_tri_ids = np.empty(total_points, dtype=np.int32)
    point_seeds = np.empty(total_points, dtype=np.float32)

    start = 0
    for tri_idx in range(M):
        count = points_per_tri[tri_idx]
        end = start + count
        point_tri_ids[start:end] = tri_idx
        # 为每个点生成伪随机种子
        point_seeds[start:end] = np.random.rand(count).astype(np.float32)
        start = end

    # 生成点
    output_np = np.zeros((total_points, 3), dtype=np.float32)
    fill_points_random_kernel(verts_np, tris_np, point_tri_ids, point_seeds, output_np)

    # 转换为 field
    point_cloud = ti.field(dtype=ti.f32, shape=(total_points, 3))
    point_cloud.from_numpy(output_np)

    return point_cloud,total_points
