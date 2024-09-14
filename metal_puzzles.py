import mlx.core as mx
from utils import MetalProblem, MetalKernel

############################################################
### Puzzle 1: Map
############################################################
# Implement a "kernel" (GPU function) that adds 10 to each 
# position of the array `a` and stores it in the array `out`. 
# You have 1 thread per position.
# 
# Note: The `source` string below is the body of your Metal kernel, the 
# function signature with be automatically generated for you. Below you'll 
# notice the `input_names` and `output_names` parameters. These define the 
# parameters for your Metal kernel.
# 
# To print out the generated Metal kernel set the environment variable `VERBOSE=1`.

def map_spec(a: mx.array):
    return a + 10

def map_test(a: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1 line)
    """

    kernel = MetalKernel(
        name="map",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Map",
    map_test,
    [a], 
    output_shape,
    grid=(SIZE,1,1), 
    spec=map_spec
)

problem.check()

############################################################
### Puzzle 2: Zip
############################################################
# Implement a kernel that takes two arrays `a` and `b`, adds each 
# element together, and stores the result in an output array `out`.
# You have 1 thread per position.

def zip_spec(a: mx.array, b: mx.array):
    return a + b

def zip_test(a: mx.array, b: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1 line)
    """

    kernel = MetalKernel(
        name="zip",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
b = mx.arange(SIZE)
output_shapes = (SIZE,)

problem = MetalProblem(
    "Zip",
    zip_test,
    [a, b],
    output_shapes,
    grid=(SIZE,1,1),
    spec=zip_spec
)

problem.check()

############################################################
### Puzzle 3: Guard
############################################################
# Implement a kernel that adds 10 to each position of `a` and 
# stores it in `out`. You have more threads than positions.

def map_guard_test(a: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1-3 lines)
    """

    kernel = MetalKernel(
        name="guard",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Guard",
    map_test,
    [a], 
    output_shape,
    grid=(8,1,1), 
    spec=map_spec
)

problem.check()

############################################################
### Puzzle 4: Map 2D
############################################################
# Implement a kernel that adds 10 to each position of `a` and 
# stores it in `out`. Input `a` is 2D and square. You have more 
# threads than positions.

def map_2D_test(a: mx.array):
    source = """
        uint thread_x = thread_position_in_grid.x;
        uint thread_y = thread_position_in_grid.y;
        // FILL ME IN (roughly 4 lines)
    """

    kernel = MetalKernel(
        name="map_2D",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 2
a = mx.arange(SIZE * SIZE).reshape((SIZE, SIZE))
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Map 2D",
    map_2D_test,
    [a], 
    output_shape,
    grid=(3,3,1), 
    spec=map_spec
)

problem.check()

############################################################
### Puzzle 5: Broadcast
############################################################
# Implement a kernel that adds `a` and `b` and stores it in `out`. 
# Inputs `a` and `b` are arrays. You have more threads than positions.

def broadcast_test(a: mx.array, b: mx.array):
    source = """
        uint thread_x = thread_position_in_grid.x;
        uint thread_y = thread_position_in_grid.y;
        // FILL ME IN (roughly 4 lines)
    """

    kernel = MetalKernel(
        name="broadcast",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 2
a = mx.arange(SIZE).reshape(SIZE, 1)
b = mx.arange(SIZE).reshape(1, SIZE)
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Broadcast",
    broadcast_test,
    [a, b], 
    output_shape,
    grid=(3,3,1), 
    spec=zip_spec
)

problem.check()

############################################################
### Puzzle 5: Broadcast
############################################################
# Implement a kernel that adds 10 to each position of `a` and 
# stores it in `out`. You have fewer threads per threadgroup 
# than the size of `a`, but more threads than positions.
#
# Note: A threadgroup is simply a group of threads within the 
# thread grid. The number of threads per threadgroup is limited 
# to a defined number, but we can have multiple different 
# threadgroups. The Metal parameter `threadgroup_position_in_grid` 
# tells us what threadgroup we are currently in.

def map_threadgroup_test(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 1-3 lines)
    """

    kernel = MetalKernel(
        name="threadgroup",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 9
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Threadgroup",
    map_threadgroup_test,
    [a], 
    output_shape,
    grid=(12,1,1), 
    threadgroup=(4,1,1),
    spec=map_spec
)

problem.check()