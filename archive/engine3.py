
import taichi as ti
import math
import sys

SCREENWIDTH=1000
SCREENHIEGHT=1000
PI = math.pi
camera_x=20.0
camera_y=20.0
camera_z=0.0
x_rotate=80.0
y_rotate=10.0
z_rotate=10.0
focal_lenth=200

ti.init(arch=ti.gpu)

Number_of_triangles=6000
MAX_Number_of_triangles=6000
dots=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3,3))
points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3,3))
#filtered_points=ti.field(dtype=ti.f32, shape=(MAX_Number_of_triangles,3))
screen_points=ti.field(dtype=ti.i32, shape=(MAX_Number_of_triangles,3,2))
#faces=ti.field(dtype=ti.f32, shape=(MAX_Number_of_dot//2,3))

screen_buffer=ti.field(dtype=ti.f32, shape=(SCREENWIDTH, SCREENHIEGHT, 3))


@ti.func
def clear():
    for i, j in ti.ndrange(SCREENWIDTH,SCREENHIEGHT):
        screen_buffer[i,j,0]=0
        screen_buffer[i,j,1]=0
        screen_buffer[i,j,2]=0

@ti.kernel
def generate_dot():

    for i in range(Number_of_triangles):   # shape[0] = 100

        dots[i,0, 0] = ti.random() * 100.0
        dots[i,0, 1] = ti.random() * 100.0
        dots[i,0, 2] = ti.random() * 100.0
        dots[i,1, 0] = ti.random() * 100.0
        dots[i,1, 1] = ti.random() * 100.0
        dots[i,1, 2] = ti.random() * 100.0
        dots[i,2, 0] = ti.random() * 100.0
        dots[i,2, 1] = ti.random() * 100.0
        dots[i,2, 2] = ti.random() * 100.0
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
    clear()

    count = 0


    for i in range(Number_of_triangles):


        # 平移
        for j in range(3):


            points[i,j, 0] = dots[i,j, 0] - camera_x
            points[i,j, 1] = dots[i,j, 1] - camera_z
            points[i,j, 2] = dots[i,j, 2] - camera_y



            # 旋转
            points[i,j, 0] , points[i,j, 2] = rotate(points[i,j, 0], points[i,j, 2],z_rotate)
            points[i,j, 0] , points[i,j, 1] = rotate(points[i,j, 0], points[i,j, 1], y_rotate)
            points[i,j, 1], points[i,j, 2] = rotate(points[i,j, 1], points[i,j, 2], x_rotate)




                # 背面剔除
        if (points[i,0, 2] <= 0.1 and ti.abs(focal_lenth * points[i,0, 0]  / points[i,0, 2]) < SCREENWIDTH / 2 and ti.abs(focal_lenth * points[i,0, 1] / points[i,0, 2]) < SCREENHIEGHT / 2) or (points[i,1, 2] <= 0.1 and ti.abs(focal_lenth * points[i,1, 0]  / points[i,1, 2]) < SCREENWIDTH / 2 and ti.abs(focal_lenth * points[i,1, 1] / points[i,1, 2]) < SCREENHIEGHT / 2) or (points[i,2, 2] <= 0.1 and ti.abs(focal_lenth * points[i,2, 0]  / points[i,2, 2]) < SCREENWIDTH / 2 and ti.abs(focal_lenth * points[i,2, 1] / points[i,2, 2]) < SCREENHIEGHT / 2):

            # 存储屏幕坐标

            for j in range(3):
                screen_points[i,j, 0] =ti.cast( focal_lenth * points[i,j, 0]  / points[i,j, 2] + SCREENWIDTH / 2,ti.i32)
                screen_points[i,j, 1] = ti.cast(focal_lenth * points[i,j, 1] / points[i,j, 2] + SCREENHIEGHT / 2,ti.i32)
        else:
            for j in range(3):
                screen_points[i,j, 0]=ti.cast(10000,ti.i32)
                screen_points[i,j, 0]=ti.cast(10000,ti.i32)



    #for i, j in ti.ndrange(SCREENWIDTH,SCREENHIEGHT):
    for p,o in ti.ndrange(Number_of_triangles,3):


        i=screen_points[p,o,0]
        j=screen_points[p,o,1]
        screen_buffer[i,j,0]=255
        screen_buffer[i,j,1]=255

        #screen_buffer[i,j,0]==255

# 创建GUI窗口，标题为 "My Renderer"，分辨率与缓冲区一致
gui = ti.GUI("My Renderer", res=(SCREENWIDTH,SCREENHIEGHT))

# 主循环
while gui.running:
    #print(visible_count[None])
    #global camera_y,camera_z,camera_x,x_rotate,y_rotate,z_rotate,focal_lenth


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


    gui.set_image(screen_buffer)  # 将数据显示到窗口
    gui.show()
