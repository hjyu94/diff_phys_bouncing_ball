import taichi as ti

real = ti.f32
ti.init(default_fp=real, flatten_if=True, debug=True)

max_steps = 512
steps = 128
dt = 0.02
lr = 0.25
elasticity = 0.8

gravity = ti.Vector([0, -0.98])

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

x = vec()
v = vec()

init_x = vec()
init_v = vec()

loss = scalar()
impulse = vec()

ball_radius = 3

gui = ti.GUI('Bouncing Ball', (360, 360))

margin = 0.01
goal = [0.8, 0.2]

def allocate_fields():
    ti.root.dense(ti.i, max_steps).place(x, v, impulse)
    ti.root.place(init_x, init_v)
    ti.root.place(loss)
    ti.root.lazy_grad()


@ti.kernel
def clear():
    for t in range(max_steps):
        impulse[t] = ti.Vector([0.0, 0.0])


@ti.kernel
def initialize():
    x[0] = init_x[None]
    v[0] = init_v[None]


# @ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t][0] - goal[0])**2 + (x[t][1] - goal[1])**2


def advance(t: ti.i32):
    # force = gravity
    
    # if x[t-1][1] < margin:
    #     normal_force = -gravity
    #     force += normal_force

    # v[t] = v[t - 1] + force * dt + impulse[t - 1]
    v[t] = v[t - 1] + impulse[t]
    x[t] = x[t - 1] + dt * v[t]


# @ti.kernel
def has_collision(t):
    return x[t][0] < margin or x[t][0] > 1 - margin or x[t][1] < margin or x[t][1] > 1 - margin


# @ti.kernel
def collide(t: ti.i32):
    if has_collision(t):
        imp = ti.Vector([0.0, 0.0])
        
        if (x[t][1] < margin and v[t][1] < 0) or (x[t][1] > 1 - margin and v[t][1] > 0):
            imp = ti.Vector([0, -(1+elasticity) * v[t][1]])
            
        elif (x[t][0] < margin and v[t][0] < 0) or (x[t][0] > 1 - margin and v[t][0] > 0):
            imp = ti.Vector([-(1+elasticity) * v[t][0], 0])
        
        impulse[t+1] += imp
        
    
def forward():
    initialize()

    for t in range(1, steps):
        collide(t-1)
        advance(t)
        
        gui.clear()
        gui.circle((x[t][0], x[t][1]), 0xCCCCCC, ball_radius)
        gui.circle((goal[0], goal[1]), 0xFFCCCC, ball_radius)
        gui.show()

    compute_loss(steps - 1)


# @ti.kernel
# def randomize():
#     init_x[None] = [0.1, 0.5]
#     init_v[None] = [ti.random(), ti.random()]
#     # init_v[None] = [1, 0]


def optimize():
    # randomize()
    init_x[None] = [0.1, 0.5]
    init_v[None] = [1, 1]
        
    for iter in range(200):
        clear()

        with ti.ad.Tape(loss):
            forward()

        print('Iter=', iter, 'Loss=', loss[None])
        for d in range(2):
            init_x[None][d] -= lr * init_x.grad[None][d]
            init_v[None][d] -= lr * init_v.grad[None][d]

    clear()
    forward()

    
if __name__ == '__main__':
    allocate_fields()
    optimize()
