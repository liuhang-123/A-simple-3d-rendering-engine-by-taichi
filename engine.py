
import taichi as ti
import math
import sys
SCREENWIDTH=1500
SCREENHEIGHT=1500
PI = math.pi
camera_x=20.0
camera_y=20.0
camera_z=0.0
x_rotate=80.0
y_rotate=10.0
z_rotate=10.0
focal_lenth=500

ti.init(arch=ti.gpu)

Number_of_points=1000000
MAX_Number_of_points=1000000
dots=ti.field(dtype=ti.f32, shape=(MAX_Number_of_points,3))
points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_points,3))
#filtered_points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_points,3))
screen_points=ti.field(dtype=ti.i32, shape=(MAX_Number_of_points,2))
#faces=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dot//2,3))
visible_count = ti.field(dtype=ti.i32, shape=())
bool_f=ti.field(dtype=ti.i8,shape=(MAX_Number_of_points))

screen_buffer=ti.field(dtype=ti.f32, shape=(SCREENWIDTH, SCREENHEIGHT, 3))
@ti.kernel
def generate_dot():

    for i in range(Number_of_points):   # shape[0] = 100

        dots[i, 0] = ti.random() * 100.0
        dots[i, 1] = ti.random() * 100.0
        dots[i, 2] = ti.random() * 100.0
        points[i, 0] = ti.random() * 100.0
        points[i, 1] = ti.random() * 100.0
        points[i, 2] = ti.random() * 100.0
generate_dot()

@ti.func
def rotate(x,y,theta):
    theta_rad = theta * PI / 180
    new_x=x*ti.cos(theta_rad)-y*ti.sin(theta_rad)
    new_y=x*ti.sin(theta_rad)+y*ti.cos(theta_rad)
    return new_x,new_y

@ti.func
def clear():
    pass
    '''for i, j in ti.ndrange(SCREENWIDTH,SCREENHEIGHT):
    screen_buffer[i,j] = ti.Vector([0, 0, 0])'''
    '''screen_buffer[i,j,0]=0
    screen_buffer[i,j,1]=0
    screen_buffer[i,j,2]=0'''

@ti.kernel
def world_to_cam(camera_x: ti.f32, camera_y: ti.f32, camera_z: ti.f32,
                 x_rotate: ti.f32, y_rotate: ti.f32, z_rotate: ti.f32,focal_lenth:ti.f32):
    screen_buffer.fill(0)
    x_rotate_=x_rotate*PI/180
    y_rotate_=y_rotate*PI/180
    z_rotate_=z_rotate*PI/180

    x_cos=ti.cos(x_rotate_)
    x_sin=ti.sin(x_rotate_)
    y_cos=ti.cos(y_rotate_)
    y_sin=ti.sin(y_rotate_)
    z_cos=ti.cos(z_rotate_)
    z_sin=ti.sin(z_rotate_)

    for i in range(Number_of_points):
        bool_f[i]=ti.cast(1,ti.i8)





        points[i, 0] = points[i, 0] - camera_x
        points[i, 1] = points[i, 1] - camera_z
        points[i, 2] = points[i, 2] - camera_y



        # 旋转
        points[i, 0],points[i, 2]  = points[i, 0]*z_cos-points[i, 2]*z_sin,points[i, 0]*z_sin+points[i, 2]*z_cos

        points[i, 0],points[i, 1]  = points[i, 0]*y_cos-points[i, 1]*y_sin,points[i, 0]*y_sin+points[i, 1]*y_cos

        points[i, 1],points[i, 2]  = points[i, 1]*x_cos-points[i, 2]*x_sin,points[i, 1]*x_sin+points[i, 2]*x_cos




                # 背面剔除
        if (points[i, 2] >= 0.1 and ti.abs(focal_lenth * points[i, 0]  / points[i, 2]) < SCREENWIDTH / 2 and ti.abs(focal_lenth * points[i, 1] / points[i, 2]) < SCREENHEIGHT / 2):


                screen_points[i, 0] =ti.cast( focal_lenth * points[i, 0] / points[i, 2] + SCREENWIDTH / 2,ti.i32)
                screen_points[i, 1] = ti.cast(focal_lenth * points[i, 1] / points[i, 2] + SCREENHEIGHT / 2,ti.i32)
                bool_f[i]=ti.cast(1,ti.i8)


        else:

            screen_points[i, 0]=ti.cast(0,ti.i32)
            screen_points[i, 1]=ti.cast(0,ti.i32)

    '''#render triangle
    for x,y in ti.ndrange(SCREENWIDTH,SCREENHEIGHT):
        for i in range(Number_of_points):
            if bool_f[i]==1:
                if point_xy_in_triangle_i(x,y,i):
                    screen_buffer[x,y,0]=225
                    break'''



    #for i, j in ti.ndrange(SCREENWIDTH,SCREENHEIGHT):
    for p in range(Number_of_points):
        #if 0<screen_points[p,0] and screen_points[p,0]<SCREENWIDTH and 0<screen_points[p,1] and screen_points[p,1]<SCREENHEIGHT:
        i=screen_points[p,0]
        j=screen_points[p,1]
        screen_buffer[i,j,0]=255
        screen_buffer[i,j,1]=255


'''
for x in range(visible_count[None]):
    print(f'{screen_points[x,0]},{screen_points[x,1]}')
'''

# pynput 控制部分
from pynput import keyboard
import threading
import sys

# 键盘状态
keys_pressed = set()
key_lock = threading.Lock()

def on_press(key):
    with key_lock:
        try:
            if hasattr(key, 'char') and key.char:
                keys_pressed.add(key.char)
        except AttributeError:
            if key == keyboard.Key.space:
                keys_pressed.add('space')
            elif key == keyboard.Key.shift:
                keys_pressed.add('shift')
            elif key == keyboard.Key.ctrl:
                keys_pressed.add('ctrl')
            # 方向键和数字键
            elif key == keyboard.Key.up:
                keys_pressed.add('up')
            elif key == keyboard.Key.down:
                keys_pressed.add('down')
            elif key == keyboard.Key.left:
                keys_pressed.add('left')
            elif key == keyboard.Key.right:
                keys_pressed.add('right')

def on_release(key):
    with key_lock:
        try:
            if hasattr(key, 'char') and key.char:
                keys_pressed.discard(key.char)
        except AttributeError:
            if key == keyboard.Key.space:
                keys_pressed.discard('space')
            elif key == keyboard.Key.shift:
                keys_pressed.discard('shift')
            elif key == keyboard.Key.ctrl:
                keys_pressed.discard('ctrl')
            # 方向键和数字键
            elif key == keyboard.Key.up:
                keys_pressed.discard('up')
            elif key == keyboard.Key.down:
                keys_pressed.discard('down')
            elif key == keyboard.Key.left:
                keys_pressed.discard('left')
            elif key == keyboard.Key.right:
                keys_pressed.discard('right')

# 启动监听器
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()

# 创建GUI窗口
gui = ti.ui.Window("Renderer", (SCREENWIDTH, SCREENHEIGHT))
canvas = gui.get_canvas()

# 速度参数
move_speed = 0.5
rotate_speed = 0.5

# 主循环
while gui.running:
    # 重置所有速度为0
    camera_x = 0.0
    camera_y = 0.0
    camera_z = 0.0
    x_rotate = 0.0
    y_rotate = 0.0
    z_rotate = 0.0

    # 处理键盘输入
    with key_lock:
        current_keys = keys_pressed.copy()

    # WASD + QE 三轴平移
    if 'w' in current_keys:
        camera_z = move_speed
    if 's' in current_keys:
        camera_z = -move_speed
    if 'a' in current_keys:
        camera_x = -move_speed
    if 'd' in current_keys:
        camera_x = move_speed
    if 'q' in current_keys:
        camera_y = -move_speed
    if 'e' in current_keys:
        camera_y = move_speed

    # 方向键控制视角旋转（俯仰和偏航）
    if 'j' in current_keys:
        x_rotate = -rotate_speed
    if 'u' in current_keys:
        x_rotate = rotate_speed
    if 'h' in current_keys:
        z_rotate = -rotate_speed
    if 'k' in current_keys:
        z_rotate = rotate_speed

    # 数字键 0 和 9 控制滚转
    if '0' in current_keys:
        y_rotate = -rotate_speed
    if '9' in current_keys:
        y_rotate = rotate_speed

    world_to_cam(
        camera_x=camera_x,
        camera_y=camera_y,
        camera_z=camera_z,
        x_rotate=x_rotate,
        y_rotate=y_rotate,
        z_rotate=z_rotate,
        focal_lenth=focal_lenth
    )

    canvas.set_image(screen_buffer)
    gui.show()

# 停止监听器
keyboard_listener.stop()

