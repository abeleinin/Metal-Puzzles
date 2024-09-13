# Metal Puzzles (WIP)

Inspired by [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) and [@awnihannun](https://x.com/awnihannun/status/1833376670063202536)!

## Coming Soon
- Documentation
- More puzzles
- `problem.show()` support

If you're interested in learning GPU programming on an Apple Silicon computer, this is a great repository for you! Whether you're new to GPU programming or have exerpience with CUDA, these puzzles provide an accessible way to learn GPU programming on Apple Silicon.

In the following exercises, you'll use the `mx.fast.metal_kernel()` function from Apple's [mlx](https://github.com/ml-explore/mlx) framework, which allows you to write custom Metal kernels through a Python/C++ API. The function takes a `source` string, which defines the Metal kernel body using the Metal Shading Language. 

If you're interested in more material, check out the [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and the [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

```sh
python -m pip install git+https://github.com/ml-explore/mlx
```

```python
import mlx.core as mx
from utils import MetalKernel, MetalProblem
```

## Puzzle 1: Map

Implement a "kernel" (GPU function) that adds 10 to each position of the array `a` and stores it in the array `out`.  You have 1 thread per position.

**Note:** The `source` string below is the body of your Metal kernel, the function signature with be automatically generated for you. Below you'll notice the `input_names` and `output_names` parameters. These define the parameters for your Metal kernel.

To print out the generated Metal kernel set the environment variable `VERBOSE=1`.

```python
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
    threadgroup=(SIZE,1,1), 
    spec=map_spec
)
```

```python
problem.check()
```

    Failed Tests.
    Yours: [0. 0. 0. 0.]
    Spec : [10 11 12 13]

## Puzzle 2: Zip 

Implement a kernel that takes two arrays `a` and `b`, adds each element together, and stores the result in an output array `out`. You have 1 thread per position.

```python
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
    threadgroup=(SIZE,1,1),
    spec=zip_spec
)
```

```python
problem.check()
```

    Failed Tests.
    Yours: [0. 0. 0. 0.]
    Spec : [0 2 4 6]

## Puzzle 3: Guard

Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have more threads than positions.

```python
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
    threadgroup=(8,1,1), 
    spec=map_spec
)
```

```python
problem.check()
```

    Failed Tests.
    Yours: [0. 0. 0. 0.]
    Spec : [10 11 12 13]