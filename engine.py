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

Number_of_triangles=100
MAX_Number_of_triangles=100
#dots=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3))
triangles=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3,3))
#filtered_triangles=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3))
screen_triangles=ti.field(dtype=ti.i32, shape=(MAX_Number_of_triangles,3,2))
#faces=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dot//2,3))
visible_count = ti.field(dtype=ti.i32, shape=())
bool_f=ti.field(dtype=ti.i8,shape=(MAX_Number_of_triangles))
#index_field=ti.field(dtype=ti.i32,shape=(MAX_Number_of_triangles//2,3))

screen_buffer=ti.field(dtype=ti.f32, shape=(SCREENWIDTH, SCREENHEIGHT, 3))
@ti.kernel
def generate_dot():

    for i,d in ti.ndrange(Number_of_triangles,3):


        triangles[i,d, 0] = ti.random() * 100.0
        triangles[i,d, 1] = ti.random() * 100.0
        triangles[i,d, 2] = ti.random() * 100.0
generate_dot()

@ti.func
def rotate(x,y,theta):
    theta_rad = theta * PI / 180
    new_x=x*ti.cos(theta_rad)-y*ti.sin(theta_rad)
    new_y=x*ti.sin(theta_rad)+y*ti.cos(theta_rad)
    return new_x,new_y

@ti.func
def point_xy_in_triangles_i(x, y, i):
    x1 = screen_triangles[i, 0, 0]
    y1 = screen_triangles[i, 0, 1]
    x2 = screen_triangles[i, 1, 0]
    y2 = screen_triangles[i, 1, 1]
    x3 = screen_triangles[i, 2, 0]
    y3 = screen_triangles[i, 2, 1]

    
    d1 = (x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)
    d2 = (x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)
    d3 = (x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)

    # 判断是否同侧（全>=0 或 全<=0）
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    has_neg = d1 < 0 or d2 < 0 or d3 < 0

    result = not (has_pos and has_neg)
    return result



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

    for i,d in ti.ndrange(Number_of_triangles,3):
        bool_f[i]=ti.cast(1,ti.i8)





        triangles[i,d, 0] = triangles[i,d, 0] - camera_x
        triangles[i,d, 1] = triangles[i,d, 1] - camera_z
        triangles[i,d, 2] = triangles[i,d, 2] - camera_y



        # 旋转
        triangles[i,d, 0],triangles[i,d, 2]  = triangles[i,d, 0]*z_cos-triangles[i,d, 2]*z_sin,triangles[i,d, 0]*z_sin+triangles[i,d, 2]*z_cos

        triangles[i,d, 0],triangles[i,d, 1]  = triangles[i,d, 0]*y_cos-triangles[i,d, 1]*y_sin,triangles[i,d, 0]*y_sin+triangles[i,d, 1]*y_cos

        triangles[i,d, 1],triangles[i,d, 2]  = triangles[i,d, 1]*x_cos-triangles[i,d, 2]*x_sin,triangles[i,d, 1]*x_sin+triangles[i,d, 2]*x_cos




        # 背面剔除
        if (triangles[i,d, 2] >= 0.1 and ti.abs(focal_lenth * triangles[i,d, 0]  / triangles[i,d, 2]) < SCREENWIDTH / 2 and ti.abs(focal_lenth * triangles[i,d, 1] / triangles[i,d, 2]) < SCREENHEIGHT / 2):
                screen_triangles[i,d, 0] =ti.cast( focal_lenth * triangles[i,d, 0] / triangles[i,d, 2] + SCREENWIDTH / 2,ti.i32)
                screen_triangles[i,d, 1] = ti.cast(focal_lenth * triangles[i,d, 1] / triangles[i,d, 2] + SCREENHEIGHT / 2,ti.i32)
        else:
           bool_f[i]=ti.cast(0,ti.i8)





        '''else:

            screen_triangles[i, 0]=ti.cast(0,ti.i32)
            screen_triangles[i, 1]=ti.cast(0,ti.i32)'''

    #render triangles

    for i in range(Number_of_triangles):
        if bool_f[i]==1:
            x1 = screen_triangles[i, 0, 0]
            y1 = screen_triangles[i, 0, 1]
            x2 = screen_triangles[i, 1, 0]
            y2 = screen_triangles[i, 1, 1]
            x3 = screen_triangles[i, 2, 0]
            y3 = screen_triangles[i, 2, 1]
            # 计算包围盒（限制在屏幕范围内）
            min_x = ti.max(0, ti.min(ti.min(x1, x2), x3))
            max_x = ti.min(SCREENWIDTH - 1, ti.max(ti.max(x1, x2), x3))
            min_y = ti.max(0, ti.min(ti.min(y1, y2), y3))
            max_y = ti.min(SCREENHEIGHT - 1, ti.max(ti.max(y1, y2), y3))

            for x,y in ti.ndrange((min_x,max_x),(min_y,max_y)):
                if point_xy_in_triangles_i(x,y,i):
                    screen_buffer[x,y,0]=225
                    screen_buffer[x,y,1]=225




    #for i, j in ti.ndrange(SCREENWIDTH,SCREENHEIGHT):
    '''for p in range(Number_of_triangles):
        if bool_f[p]==1:

            i=screen_triangles[p,0]
            j=screen_triangles[p,1]
            screen_buffer[i,j,0]=255
            screen_buffer[i,j,1]=255'''


'''
for x in range(visible_count[None]):
    print(f'{screen_triangles[x,0]},{screen_triangles[x,1]}')
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
    act=False
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
    #if current_keys:

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
    if current_keys:
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
