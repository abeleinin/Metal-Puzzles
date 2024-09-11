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
        inputs, source = self.fn(*self.inputs)

        kernel = mx.fast.metal_kernel(
            name=self.name,
            source=source,
        )

        outputs = kernel(
            inputs=inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes={"out": self.output_shapes},
            output_dtypes={"out": mx.float32},
        )

        return outputs["out"]

    def check(self):
        x = self.run_metal()
        y = self.spec(*self.inputs)

        if mx.allclose(x, y): 
            print("Passed Tests!")
        else:
            print("Failed Tests.")
            print("Yours:", x)
            print("Spec :", y)