import taichi as ti
import random
ti.init(arch=ti.gpu)
gui=ti.GUI("3d_point_cloud_render",(640,640))
Number_of_dots=100
dots=ti.field(dtype=ti.f32, shape=(Number_of_dots,3))


@ti.kernel
def generate_dot():
    for i in range(dots.shape[0]):   # shape[0] = 100

        dots[i, 0] = ti.random() * 100.0
        dots[i, 1] = ti.random() * 100.0
        dots[i, 2] = ti.random() * 100.0
generate_dot()

@ti.func
def cartesian_to_polar(x: ti.f32, y: ti.f32):
    """
    平面坐标 -> 极坐标 (角度制)
    返回: (r, theta_deg) 其中 theta_deg 范围 [-180, 180]
    """
    r = ti.sqrt(x * x + y * y)
    theta_rad = ti.atan2(y, x)
    theta_deg = theta_rad * 180.0 / math.pi
    return r, theta_deg

@ti.func
def polar_to_cartesian(r: ti.f32, theta_deg: ti.f32):
    """
    极坐标 (角度制) -> 平面坐标
    """
    theta_rad = theta_deg * math.pi / 180.0
    x = r * ti.cos(theta_rad)
    y = r * ti.sin(theta_rad)
    return x, y


@ti.func
def plane_rotate(x,y,theta):
    new_x = x·cos(theta) - y·sin(theta)
    new_y = x·sin(theta) + y·cos(theta)
    return new_x,new_y

@ti.func
def world_to_cam():



@ti.func
def z_check():

@ti.kernel
def render():
    for i in range(dots.shape[0]):






while gui.running:

    gui.show()
