import os

from dataclasses import dataclass
from typing import List, Tuple, Any

import mlx.core as mx

@dataclass
class MetalProblem:
    name: str
    fn: Any
    inputs: List[mx.array]
    output_shapes: Tuple[int]
    grid: Tuple[int] = (1,1,1)
    threadgroup: Tuple[int] = (1,1,1)
    spec: Any = None

    def run_metal(self):
        assert mx.metal.is_available(), "Metal is not available"

        kernel = self.fn(*self.inputs)

        outputs = kernel(
            inputs=self.inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[self.output_shapes],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
            verbose=os.getenv("VERBOSE")=='1',
        )

        return outputs[0]

    def check(self):
        try:
            x = self.run_metal()
            y = self.spec(*self.inputs)

            if mx.allclose(x, y): 
                print("Passed Tests!")
                return 

            print("Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)

        except AssertionError as e:
            print(f"Error: {e}")
