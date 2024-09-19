import os
import re

from dataclasses import dataclass
from typing import List, Tuple, Any

import mlx.core as mx

@dataclass
class MetalKernel:
    name: str
    input_names: List[str]
    output_names: List[str]
    header: str = ""
    source: str = ""

    def __call__(self):
        return mx.fast.metal_kernel(
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
            header=self.header,
            source=self.source,
        )

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

        outputs = self.metalKernel()(
            inputs=self.inputs,
            grid=self.grid,
            threadgroup=self.threadgroup,
            output_shapes=[self.output_shapes],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
            verbose=os.getenv("VERBOSE")=='1',
            init_value=0,
        )

        return outputs[0]
    
    def score(self):
        inputs = {}
        for i in range(len(self.inputs)):
            curr = self.inputs[i]
            name = self.metalKernel.input_names[i]
            inputs[name] = TrackedArray(name, curr.reshape(curr.size))
            inputs[name + "_shape"] = curr.shape
            inputs[name + "_ndim"] = curr.ndim
            inputs[name + "_strides"] = mx.array([curr.shape[0], 1])
        
        outputs = {}
        for name in self.metalKernel.output_names:
            temp = mx.zeros(self.output_shapes)
            outputs[name] = TrackedArray(name, temp.reshape(temp.size))

        locals().update(inputs)
        locals().update(outputs)

        threads_per_threadgroup_x = self.threadgroup[0]
        threads_per_threadgroup_y = self.threadgroup[1]
        threads_per_threadgroup_z = self.threadgroup[2]

        full = {'in_reads' : 0, 'out_writes' : 0, 'shared_reads' : 0, 'shared_writes' : 0}
        metal_py = convert_source_to_py(self.metalKernel.header + self.metalKernel.source)
        for grid_x in range(self.grid[0]):
            for grid_y in range(self.grid[1]):
                thread_position_in_grid_x, thread_position_in_grid_y = grid_x, grid_y

                threadgroup_position_in_grid_x = grid_x // self.threadgroup[0]
                thread_position_in_threadgroup_x = grid_x % self.threadgroup[0]
                threadgroup_position_in_grid_y = grid_y // self.threadgroup[1]
                thread_position_in_threadgroup_y = grid_y % self.threadgroup[1]

                exec(metal_py)

                local_variables = locals().copy()
                for name, value in local_variables.items():
                    if isinstance(value, ThreadgroupMemory):
                        full['shared_reads'] = max(locals()[name].get_reads(), full['shared_reads'])
                        full['shared_writes'] = max(locals()[name].get_writes(), full['shared_writes'])
                    if isinstance(value, TrackedArray):
                        locals()[name].set_max()

        for name in self.metalKernel.input_names:
            full['in_reads'] += locals()[name].get_reads()
        
        full['out_writes'] = locals()[self.metalKernel.output_names[0]].get_writes()
        return (f"""# {self.name}

    Score (Max Per Thread):
    | {'Global Reads':>13} | {'Global Writes':>13} | {'Shared Reads' :>13} | {'Shared Writes' :>13} |
    | {full['in_reads']:>13} | {full['out_writes']:>13} | {full['shared_reads']:>13} | {full['shared_writes']:>13} | 
        """) 
    
    def show(self):
        self.metalKernel = self.fn(*self.inputs)
        print(self.score())

    def check(self):
        try:
            self.metalKernel = self.fn(*self.inputs)

            if os.getenv("MTL_CAPTURE_ENABLED") == '1':
                mx.eval(*self.inputs)
                
                traceName = f"custom_kernel_{self.metalKernel.name}.gputrace"
                mx.metal.start_capture(traceName)
                for _ in range(2): mx.eval(self.run_metal())
                mx.metal.stop_capture()

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

class TrackedArray:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.reads_per_thread = 0
        self.writes_per_thread = 0
        self.max_reads = 0
        self.max_writes = 0
    
    def __getitem__(self, index):
        if (index > len(self.data)): 
            raise Exception(f"Out-of-bounds access in mx.array '{self.name}' at index {index}.")
        self.reads_per_thread += 1
        return self.data[index]
    
    def __setitem__(self, index, value):
        if (index > len(self.data)): 
            raise Exception(f"Out-of-bounds access in mx.array '{self.name}' at index {index}.")
        self.writes_per_thread += 1
        self.data[index] = value
    
    def get_reads(self):
        return self.max_reads
    
    def get_writes(self):
        return self.max_writes
    
    def set_max(self):
        self.max_reads = max(self.reads_per_thread, self.max_reads)
        self.max_writes = max(self.writes_per_thread, self.max_writes)
        self.reads_per_thread = 0
        self.writes_per_thread = 0

class ThreadgroupMemory:
    def __init__(self, size, size2=0):
        shape = (size, size2) if size2 != 0 else size
        self.data = mx.zeros(shape)
        self.reads_per_thread = 0
        self.writes_per_thread = 0
    
    def __getitem__(self, index):
        if (index > len(self.data)): 
            raise Exception(f"Out-of-bounds access in ThreadgroupMemory at index {index}.")
        self.reads_per_thread += 1
        return self.data[index]

    def __setitem__(self, index, value):
        if (index > len(self.data)): 
            raise Exception(f"Out-of-bounds access in ThreadgroupMemory at index {index}.")
        self.writes_per_thread += 1
        self.data[index] = value
    
    def get_reads(self):
        return self.reads_per_thread
    
    def get_writes(self):
        return self.writes_per_thread

def convert_source_to_py(source):

    metal_source = preprocess_source(source)

    output_lines = []
    for_loop_incr = []
    indent_level = 0
    statement = False
    lines = metal_source.splitlines()
    for line in lines:
        line = line.strip()

        for keyword, pattern, replacement in [
            ('else if', r'\s*\}?\s*else if\s*\((.*?)\)\s*\{?', r'elif \1:'),
            ('if', r'if\s*\((.*?)\)\s*\{?', r'if \1:'),
            ('while', r'while\s*\((.*?)\)\s*\{?', r'while \1:'),
            ('else', r'\s*\}?\s*else\s*\{?', r'else:')
        ]:
            if keyword in line:
                line = re.sub(pattern, replacement, line)
                line = '    ' * indent_level + line
                indent_level += 1
                statement = True

        m = re.match(r'for\s*\(\s*(.*?);\s*(.*?);\s*(.*?)\s*\)\s*\{?', line)
        if m:
            init = m.group(1).strip()
            cond = m.group(2).strip()
            incr = m.group(3).strip()

            output_lines.append('    ' * indent_level + init)
            output_lines.append('    ' * indent_level + "while " + cond + ":")

            if '++' in incr:
                incr = re.sub(r'(\+\+)(\w+)', r'\2 += 1', incr)
                incr = re.sub(r'(\w+)(\+\+)', r'\1 += 1', incr)
            elif '--' in incr:
                incr = re.sub(r'(\-\-)(\w+)', r'\2 -= 1', incr)
                incr = re.sub(r'(\w+)(\-\-)', r'\1 -= 1', incr)
            indent_level += 1
            for_loop_incr.append('    ' * indent_level + incr)
            continue

        if line == '{':
            indent_level += 1
            continue
        elif line == '}':
            for incr in for_loop_incr:
                output_lines.append(incr)
                for_loop_incr.pop()
            indent_level -= 1
            continue

        if not statement:
            line = line.replace(';', '')
            line = '    ' * indent_level + line
        output_lines.append(line)
        statement = False

    return '\n'.join(output_lines)

def preprocess_source(source):
    metal_source = re.sub(r'//.*', '', source)
    metal_source = re.sub(r'threadgroup float (\w+)\[(\w+)\]\[(\w+)\];', r'\1 = ThreadgroupMemory(\2, \3)', metal_source)
    metal_source = re.sub(r'threadgroup float (\w+)\[(\w+)\];', r'\1 = ThreadgroupMemory(\2)', metal_source)
    metal_source = re.sub(r'threadgroup_barrier\(mem_flags::mem_threadgroup\);', '', metal_source)
    metal_source = re.sub(r'metal::', '', metal_source)
    metal_source = re.sub(r'\b(uint|int|float|double|auto|constant)\b', '', metal_source)
    metal_source = metal_source.replace('&&', 'and')
    metal_source = metal_source.replace('||', 'or')

    for axis in ["x", "y", "z"]:
        replacements = [
            (r'thread_position_in_grid\.', 'thread_position_in_grid_'),
            (r'threadgroup_position_in_grid\.', 'threadgroup_position_in_grid_'),
            (r'threads_per_threadgroup\.', 'threads_per_threadgroup_'),
            (r'thread_position_in_threadgroup\.', 'thread_position_in_threadgroup_')
        ]
        for old, new in replacements:
            metal_source = re.sub(old+axis, new+axis, metal_source)
    
    return metal_source 
