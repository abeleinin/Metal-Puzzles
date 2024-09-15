# Metal Puzzles

Port of [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) to [Metal](https://en.wikipedia.org/wiki/Metal_(API)) using [MLX custom_kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). Inspired by [@awnihannun](https://x.com/awnihannun/status/1833376670063202536)!

![Metal Puzzles Logo](/metal_puzzles.png)

If you're interested in learning GPU programming on an Apple Silicon computer, this is a great repository for you! Whether you're new to GPU programming or have exerpience with CUDA, these puzzles provide an accessible way to learn GPU programming on Apple Silicon.

In the following exercises, you'll use the `mx.fast.metal_kernel()` function from Apple's [mlx](https://github.com/ml-explore/mlx) framework, which allows you to write custom Metal kernels through a Python/C++ API. The function takes a `source` string, which defines the Metal kernel body using the Metal Shading Language. 

If you're interested in more material, check out the [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and the [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

```sh
python -m pip install mlx
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
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0], dtype=float32)
Spec : array([10, 11, 12, 13], dtype=int32)
```

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
    spec=zip_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0], dtype=float32)
Spec : array([0, 2, 4, 6], dtype=int32)
```

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
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0], dtype=float32)
Spec : array([10, 11, 12, 13], dtype=int32)
```

## Puzzle 4: Map 2D

Implement a kernel that adds 10 to each position of `a` and stores it in `out`. Input `a` is 2D and square. You have more threads than positions.

```python
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
output_shape = (SIZE,SIZE)

problem = MetalProblem(
    "Map 2D",
    map_2D_test,
    [a], 
    output_shape,
    grid=(3,3,1), 
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([[0, 0],
       [0, 0]], dtype=float32)
Spec : array([[10, 11],
       [12, 13]], dtype=int32)
```

## Puzzle 5: Broadcast

Implement a kernel that adds `a` and `b` and stores it in `out`. Inputs `a` and `b` are arrays. You have more threads than positions.

```python
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
output_shape = (SIZE,SIZE)

problem = MetalProblem(
    "Broadcast",
    broadcast_test,
    [a, b], 
    output_shape,
    grid=(3,3,1), 
    spec=zip_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([[0, 0],
       [0, 0]], dtype=float32)
Spec : array([[0, 1],
       [1, 2]], dtype=int32)
```

## Puzzle 6: Threadgroups

Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have fewer threads per threadgroup than the size of `a`, but more threads than positions.

**Note:** A threadgroup is simply a group of threads within the thread grid. The number of threads per threadgroup is limited to a defined number, but we can have multiple different threadgroups. The Metal parameter `threadgroup_position_in_grid` tells us what threadgroup we are in.

```python
def map_threadgroup_test(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 1-3 lines)
    """

    kernel = MetalKernel(
        name="threadgroups",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 9
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Threadgroups",
    map_threadgroup_test,
    [a], 
    output_shape,
    grid=(12,1,1), 
    threadgroup=(4,1,1),
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
Spec : array([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=int32)
```

## Puzzle 7: Threadgroups 2D

Implement the same kernel in 2D. You have fewer threads per threadgroup than the size of `a` in both directions, but more threads than positions in the grid.

```python
def map_threadgroup_2D_test(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 5 lines)
    """

    kernel = MetalKernel(
        name="threadgroups_2D",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 5
a = mx.ones((SIZE, SIZE))
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Threadgroups 2D",
    map_threadgroup_2D_test,
    [a], 
    output_shape,
    grid=(6,6,1), 
    threadgroup=(3,3,1),
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([[0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]], dtype=float32)
Spec : array([[11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [11, 11, 11, 11, 11]], dtype=float32)
```

## Puzzle 8: Threadgroup Memory

Implement a kernel that adds 10 to each position of `a` and stores it in `out`. You have fewer threads per threadgroup than the size of `a`.

**Warning**: Each threadgroup can only have a constant amount of threadgroup memory that the threads can read and write to. After writing to threadgroup memory, you need to call `threadgroup_barrier(mem_flags::mem_threadgroup)` to ensure that threads are synchronized.

For more information read section [4.4 Threadgroup Address Space](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=86) and section [6.9 Synchronization and SIMD-Group Functions](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=177) in the Metal Shading Language Specification.

(This example does not really need threadgroup memory or synchronization, but it is a demo.)

```python
def shared_test(a: mx.array):
    source = """
        threadgroup float shared[4];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;

        if (i < a_shape[0]) {
            shared[local_i] = a[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // FILL ME IN (roughly 1-3 lines)
    """

    kernel = MetalKernel(
        name="threadgroup_memory",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 8
a = mx.ones(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Threadgroup Memory",
    shared_test,
    [a], 
    output_shape,
    grid=(SIZE,1,1), 
    threadgroup=(4,1,1),
    spec=map_spec
)
```

```python
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
Spec : array([11, 11, 11, 11, 11, 11, 11, 11], dtype=float32)
```

## Puzzle 9: Pooling

Implement a kernel that sums together the last 3 position of `a` and stores it in `out`. You have 1 thread per position. 

`threadgroup` memory is often faster than sharing data in `device` memory because it is locaed cloaser the the GPU's compute units. Be careful of uncessary reads and writes from global parameters (`a` and `out`) since their data is stored in `device` memory. You only need 1 global read and 1 global write per thread.

Tip: Remember to be careful about syncing.

```python
def pooling_spec(a: mx.array):
    out = mx.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out

def pooling_test(a: mx.array):
    source = """
        threadgroup float shared[8];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 11 lines)
    """

    kernel = MetalKernel(
        name="pooling",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 8
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Pooling",
    pooling_test,
    [a], 
    output_shape,
    grid=(SIZE,1,1), 
    threadgroup=(SIZE,1,1),
    spec=pooling_spec
)
```

```
problem.check()
```

```
Failed Tests.
Yours: array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float32)
Spec : array([0, 1, 3, 6, 9, 12, 15, 18], dtype=float32)
```

## TODO
- [ ] Add all puzzles
- [ ] Metal Debugger Tutorial Section
