from utils import MetalProblem
import mlx.core as mx

### Puzzle 1: Map

def map_spec(a: mx.array):
    return a + 10

def map_test(a: mx.array):
    inputs = {"a": a}

    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1 line)
    """

    return inputs, source 

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem("Map", map_test, [a], output_shape, grid=(SIZE,1,1), threadgroup=(SIZE,1,1), spec=map_spec)

problem.check()
