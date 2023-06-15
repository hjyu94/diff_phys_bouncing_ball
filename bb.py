import taichi as ti
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, flatten_if=True, debug=True)

max_steps = 1024
steps = 512
dt = 0.02
lr = 0.25
elasticity = 0.8
epoch = 1

gravity = ti.Vector([0, -0.98])

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

x = ti.Vector.field(2, dtype=real, shape=max_steps, needs_grad = True)
v = ti.Vector.field(2, dtype=real, shape=max_steps, needs_grad = True)

init_x = ti.Vector.field(2, dtype=real, shape=(), needs_grad = True)
init_v = ti.Vector.field(2, dtype=real, shape=(), needs_grad = True)

loss = ti.field(dtype=real, shape=(), needs_grad = True)
impulse = ti.Vector.field(2, dtype=real, shape=max_steps, needs_grad = True)

margin = 0.01
goal = [0.8, 0.2]
ball_radius = 3


gui = ti.GUI('Bouncing Ball', (360, 360))


lines = [
    [(0.01, 0.01), (0.99, 0.01)], 
    [(0.01, 0.99), (0.99, 0.99)], 
    [[0.1, 0.2], [0.3, 0.2]], 
    [[0.6, 0.5], [0.8, 0.5]], 
    [[0.3, 0.8], [0.5, 0.8]], 
    [[0.01, 0.01], [0.01, 0.99]], 
    [[0.99, 0.01], [0.99, 0.99]], 
]


@ti.kernel
def clear():
    for t in range(max_steps):
        impulse[t] = ti.Vector([0.0, 0.0])


@ti.kernel
def initialize():
    x[0] = init_x[None]
    v[0] = init_v[None]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t][0] - goal[0])**2 + (x[t][1] - goal[1])**2


@ti.kernel
def advance(t: ti.i32):
    v[t] = v[t - 1] + impulse[t]
    x[t] = x[t - 1] + dt * v[t]

epsilon = 0.03

@ti.kernel
def collide(t: ti.i32):
    imp = ti.Vector([0.0, 0.0])
    
    # horizontal
    if (x[t][1] < margin and v[t][1] < 0) or (x[t][1] > 1 - margin and v[t][1] > 0):
        imp = ti.Vector([0, -(1+elasticity) * v[t][1]])
        
    elif (x[t][0] > 0.1 and x[t][0] < 0.3) and (
        (0.2 < x[t][1] < 0.2 + epsilon and v[t][1] < 0) 
        or (0.2 - epsilon < x[t][1] < 0.2 and v[t][1] > 0)
        ):
        imp = ti.Vector([0, -(1+elasticity) * v[t][1]])
        
    elif (x[t][0] > 0.6 and x[t][0] < 0.8) and (
        (0.5 < x[t][1] < 0.5 + epsilon and v[t][1] < 0) 
        or (0.5 - epsilon < x[t][1] < 0.5 and v[t][1] > 0)
        ):
        imp = ti.Vector([0, -(1+elasticity) * v[t][1]])
        
    elif (x[t][0] > 0.3 and x[t][0] < 0.5) and (
        (0.8 < x[t][1] < 0.8 + epsilon and v[t][1] < 0) 
        or (0.8 - epsilon < x[t][1] < 0.8 and v[t][1] > 0)
        ):
        imp = ti.Vector([0, -(1+elasticity) * v[t][1]])
        
    # vertical
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
        gui.circle((goal[0], goal[1]), 0xFFCCCC, ball_radius * 1.4)
        
        for line in lines:
            gui.line(line[0], line[1], radius=1, color=0xD9D2E9)

        gui.show()

    compute_loss(steps - 1)


@ti.kernel
def randomize():
    init_x[None] = [0.4, 0.5]
    init_v[None] = [ti.random(), ti.random()]


def optimize():
    randomize()
    
    iters = []
    losses = []
    for iter in range(epoch):
        clear()
        print('init_x:',init_x)
        print('init_v:',init_v)

        with ti.ad.Tape(loss):
            forward()

        print('Iter=', iter, 'Loss=', loss[None])
        
        for d in range(2):
            init_x[None][d] -= lr * init_x.grad[None][d]
            init_v[None][d] -= lr * init_v.grad[None][d]
        
        for d in range(2):
            init_x[None][d] = min(init_x[None][d], 0.8)
            init_x[None][d] = max(init_x[None][d], 0.2)
            
        losses.append(loss[None])
            
    plt.plot(losses)
    plt.show()
        
    clear()
    forward()

    
if __name__ == '__main__':
    optimize()
