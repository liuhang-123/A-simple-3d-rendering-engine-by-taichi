import taichi as ti
import math
X_RANGE=60.0
Y_RANGE=60.0
PI = math.pi
camera_x=10.0
camera_y=10.0
camera_z=10.0
x_rotate=10.0
y_rotate=20.0
z_rotate=90.0


ti.init(arch=ti.gpu)

Number_of_dots=10
dots=ti.field(dtype=ti.f32, shape=(Number_of_dots,3))
points=ti.field(dtype=ti.f32, shape=(Number_of_dots,3))
filtered_points=ti.field(dtype=ti.f32, shape=(Number_of_dots,3))
screen_points=ti.field(dtype=ti.f32, shape=(Number_of_dots,2))



#保持角度在0-360
@ti.func
def mod(deg:ti.f32):
    scale = ti.f32(100000000.0)  # 10^8
    scaled = deg * scale
    int_scaled = ti.i32(scaled)  # 转换为整数
    int_mod = int_scaled % 360  # 整数取模
    result = ti.f32(int_mod) / scale
    return result

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
    theta_deg = theta_rad * 180.0 / PI
    return r, theta_deg

@ti.func
def polar_to_cartesian(r: ti.f32, theta_deg: ti.f32):
    """
    极坐标 (角度制) -> 平面坐标
    """
    theta_rad = theta_deg * PI / 180.0
    x = r * ti.cos(theta_rad)
    y = r * ti.sin(theta_rad)
    return x, y



@ti.kernel
def world_to_cam():
    '''
    if x_rotate>=360:
        x_rotate=mod(x_rotate)
    if y_rotate>=360:
        y_rotate=mod(y_rotate)
    if z_rotate>=360:
        z_rotate=mod(z_rotate)'''
    for i in range(dots.shape[0]):
        points[i,0]=dots[i,0]-camera_x
        points[i,1]=dots[i,1]-camera_y
        points[i,2]=dots[i,2]-camera_z

    for i in range(dots.shape[0]):

        zr,ztheta=cartesian_to_polar(points[i,0],points[i,1])
        ztheta=ztheta+z_rotate
        points[i,0],points[i,1]=polar_to_cartesian(zr,ztheta)

        yr,ytheta=cartesian_to_polar(points[i,0],points[i,2])
        ytheta=ytheta+y_rotate
        points[i,0],points[i,2]=polar_to_cartesian(yr,ytheta)

        xr,xtheta=cartesian_to_polar(points[i,1],points[i,2])
        xtheta=xtheta+x_rotate
        points[i,1],points[i,2]=polar_to_cartesian(xr,xtheta)

    len_of_filtered_point=0
    for i in range(dots.shape[0]):
        if points[i,2]<2 or points[i,2]>50:

            pass
        elif points[i,0]/points[i,2]>X_RANGE or points[i,1]/points[i,2]>Y_RANGE:
            pass
        else:
            len_of_filtered_point=len_of_filtered_point+1
            filtered_points[len_of_filtered_point,0]=points[i,0]
            filtered_points[len_of_filtered_point,1]=points[i,1]
            filtered_points[len_of_filtered_point,2]=points[i,2]
    for i in range(len_of_filtered_point):
        x=filtered_points[i,0]/filtered_points[i,2]
        y=filtered_points[i,1]/filtered_points[i,2]
        screen_points[i,0]=x
        screen_points[i,1]=y


@ti.kernel
def normalize_coords():
    for i in range(Number_of_dots):
        screen_points[i,0] = screen_points[i,0] / X_RANGE
        screen_points[i,1] = screen_points[i,1] / Y_RANGE




world_to_cam()
normalize_coords()

for i in range(screen_points.shape[0]):
    print(f"dots[{i}] = {screen_points[i,0]}")




'''

window = ti.ui.Window("Points", (800, 600), vsync=False)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))

while window.running:
    canvas.circles(screen_points, radius=5.0, color=(1, 1, 1))  # 改用 circles
    window.show()
'''
