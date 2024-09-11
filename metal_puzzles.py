import mlx.core as mx
from utils import MetalProblem

### Puzzle 1: Map

def map_spec(a: mx.array):
    return a + 10

def map_test(a: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1 line)
    """

    kernel = mx.fast.metal_kernel(
        name="map",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem("Map", map_test, [a], output_shape, grid=(SIZE,1,1), threadgroup=(SIZE,1,1), spec=map_spec)

problem.check()
