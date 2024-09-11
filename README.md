# Metal Puzzles (WIP)

Inspired by [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) and [@awnihannun](https://x.com/awnihannun/status/1833376670063202536)!

## Coming Soon
- Documentation
- More puzzles
- `problem.show()` support

## Puzzle 1: Map

```python
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
```

```python
problem.check()
```

    Failed Tests.
    Yours: [0. 0. 0. 0.]
    Spec : [10 11 12 13]
