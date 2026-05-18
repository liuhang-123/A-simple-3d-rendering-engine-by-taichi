# A-simple-3d-rendering-engine-by-taichi
2026/4/7,not started yet

2026/4/21 It successfully projected 3d points cloud to plane coordinates, It can handle translation and rotation correctly. To test whether the 3d rotation and translation were correctly handled I used matplotlib to plot the plane coordinates.

2026/4/28 I reorganized structure of data for future rendering of triangles. Replace the matplotlib  visual debug screen by taichi gui

2026/5/6 reorganized structure of data back to point rendering, fixed circular index bug. **1 million points** rendered at **150+ FPS** on a **1500^2 pixel screen** on my **5000RMB ultrabook** (engine5.py)

2026/5/** basic triangle rendering. (archive/engine6.py)

2026/5/18 render triangles in tiles so that large triangles can be rendered faster. (main/engine.py)
