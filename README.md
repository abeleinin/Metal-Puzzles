# Metal Puzzles (WIP)

Inspired by [srush](https://github.com/srush)'s [GPU Puzzles](https://github.com/srush/GPU-Puzzles) and [@awnihannun](https://x.com/awnihannun/status/1833376670063202536)!

## Coming Soon
- Documentation
- More puzzles
- `problem.show()` support

## Puzzle 1: Map

```python
def map_spec(a: mx.array):
    return a + 10

def map_metal(a: mx.array):
    inputs = {"a": a}

    source = """
        uint local_i = thread_position_in_grid.x;
        // FILL ME IN (roughly 1 line)
    """

    return inputs, source 

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem("Map", map_metal, [a], output_shape, grid=(SIZE,1,1), threadgroup=(SIZE,1,1), spec=map_spec)
```

```python
problem.check()
```

    Failed Tests.
    Yours: [0. 0. 0. 0.]
    Spec : [10 11 12 13]
