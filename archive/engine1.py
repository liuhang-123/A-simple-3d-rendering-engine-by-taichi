
import taichi as ti
import math
import pygame
import sys
pygame.init()
SCREENWIDTH=500
SCREENHIEGHT=500
PI = math.pi
camera_x=20.0
camera_y=20.0
camera_z=0.0
x_rotate=80.0
y_rotate=10.0
z_rotate=10.0
focal_lenth=200

ti.init(arch=ti.gpu)

Number_of_dots=10000
MAX_Number_of_dots=10000
dots=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dots,3))
points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dots,3))
#filtered_points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dots,3))
screen_points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dots,2))
#faces=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dot//2,3))
visible_count = ti.field(dtype=ti.i32, shape=())

@ti.kernel
def generate_dot():

    for i in range(Number_of_dots):   # shape[0] = 100

        dots[i, 0] = ti.random() * 100.0
        dots[i, 1] = ti.random() * 100.0
        dots[i, 2] = ti.random() * 100.0
generate_dot()

@ti.func
def rotate(x,y,theta):
    theta_rad = theta * PI / 180
    new_x=x*ti.cos(theta_rad)-y*ti.sin(theta_rad)
    new_y=x*ti.sin(theta_rad)+y*ti.cos(theta_rad)
    return new_x,new_y


@ti.kernel
def world_to_cam(camera_x: ti.f32, camera_y: ti.f32, camera_z: ti.f32,
                 x_rotate: ti.f32, y_rotate: ti.f32, z_rotate: ti.f32,focal_lenth:ti.f32):
    visible_count[None] = 0
    count = 0


    for i in range(Number_of_dots):
        # 平移
        points[i, 0] = dots[i, 0] - camera_x
        points[i, 1] = dots[i, 1] - camera_z
        points[i, 2] = dots[i, 2] - camera_y

        # 旋转
        points[i, 0] , points[i, 2] = rotate(points[i, 0], points[i, 2],z_rotate)
        points[i, 0] , points[i, 1] = rotate(points[i, 0], points[i, 1], y_rotate)
        points[i, 1], points[i, 2] = rotate(points[i, 1], points[i, 2], x_rotate)




        # 背面剔除
        if points[i, 2] <= 0.1:
            continue


        elif ti.abs(focal_lenth * points[i, 0]  / points[i, 2]) > SCREENWIDTH / 2 or ti.abs(focal_lenth * points[i, 1] / points[i, 2]) > SCREENHIEGHT / 2:
            continue
        else:

            # 存储屏幕坐标
            idx = ti.atomic_add(count, 1)

            screen_points[idx, 0] = focal_lenth * points[i, 0]  / points[i, 2] + SCREENWIDTH / 2
            screen_points[idx, 1] = focal_lenth * points[i, 1] / points[i, 2] + SCREENHIEGHT / 2
    visible_count[None] = count

world_to_cam(camera_x=0.0,
camera_y=0.0,
camera_z=0.0,
x_rotate=80.0,
y_rotate=180.0,
z_rotate=180.0,
focal_lenth=200)

print(visible_count[None])







'''
for x in range(visible_count[None]):
    print(f'{screen_points[x,0]},{screen_points[x,1]}')
'''



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#screen_points = ti.field(dtype=ti.f32, shape=(MAX_Number_of_dots, 2))

# 设置图形 - 添加坐标范围
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 640)   # 根据你的相机分辨率设置
ax.set_ylim(0, 480)
scatter = ax.scatter([], [], s=20)



# 更新函数
import time
timing=time.time()
def update(frame):
    global camera_y,camera_z,camera_x,x_rotate,y_rotate,z_rotate,focal_lenth


    #y_rotate+= 1
    x_rotate+=1
    #z_rotate+=1


    world_to_cam(camera_x=camera_x,
camera_y=camera_y,
camera_z=camera_z,
x_rotate=x_rotate,
y_rotate=y_rotate,
z_rotate=z_rotate,
focal_lenth=focal_lenth)
    global timing
    inter=timing-time.time()
    timing=time.time()
    print(inter)
    points_ = screen_points.to_numpy()
    points_=points_[:visible_count[None]]
    #print(points_)
    scatter.set_offsets(points_)
    return scatter,

# 动画 - 如果还是不行，把 blit 改为 False
ani = FuncAnimation(fig, update, interval=0, blit=False)  # 或 blit=False
plt.show()

