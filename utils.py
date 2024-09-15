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
    source: str

    def __call__(self):
        return mx.fast.metal_kernel(
            name=self.name,
            input_names=self.input_names,
            output_names=self.output_names,
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
        )

        return outputs[0]
    
    def verify_metal_source(self):
        inputs = {}
        for i in range(len(self.inputs)):
            curr = self.inputs[i]
            inputs[self.metalKernel.input_names[i]] = curr.reshape(curr.size).tolist()
            inputs[self.metalKernel.input_names[i] + "_shape"] = curr.shape
        
        outputs = {}
        for i in range(len(self.metalKernel.output_names)):
            out = mx.zeros(self.output_shapes)
            outputs[self.metalKernel.output_names[i]] = out.reshape(out.size)

        verify_source(self.metalKernel.source, inputs, outputs, self.grid)

    def check(self):
        try:
            self.metalKernel = self.fn(*self.inputs)

            if self.name in ["Map", "Zip", "Guard", "Map 2D", "Broadcast"]: 
                self.verify_metal_source()

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


def preprocess_code(source):
    code_lines = source.strip().split('\n')
    processed_lines = []
    for line in code_lines:
        line = line.strip()
        if re.match(r'^//', line): continue # Skip comments
        line = re.sub(r'\b(uint|int|float|double)\b', '', line) # Remove type declarations
        line = line.replace('thread_position_in_grid.x', 'thread_position_in_grid_x')
        line = line.replace('thread_position_in_grid.y', 'thread_position_in_grid_y')
        line = line.replace('thread_position_in_grid.z', 'thread_position_in_grid_z')
        processed_lines.append(line)
    return '\n'.join(processed_lines)

def evaluate_expression(expr, variables):
    # Replace array access opertator[]
    pattern = r'(\w+)\[(.+?)\]'
    def replace_array_access(match):
        array_name = match.group(1)
        index_expr = match.group(2)
        index_value = evaluate_expression(index_expr, variables)
        array = variables.get(array_name)
        if array is None:
            raise Exception(f"Array '{array_name}' not found.")
        if not isinstance(index_value, int):
            raise Exception(f"Index '{index_expr}' evaluated to non-integer value.")
        if index_value < 0 or index_value >= len(array):
            raise Exception(f"Out-of-bounds access in mx.array '{array_name}' at index {index_value}.")
        return str(array[index_value])
    expr = re.sub(pattern, replace_array_access, expr)
    
    # Replace variables with their values
    for var in sorted(variables, key=lambda v: -len(v)):  # Longer names first
        expr = re.sub(r'\b' + re.escape(var) + r'\b', str(variables[var]), expr)
    
    # Evaluate the expression safely
    try:
        return eval(expr, {"__builtins__": None}, {})
    except Exception as e:
        raise Exception(f"Error evaluating expression '{expr}': {e}")

def process_assignment(line, variables):
    line = line.rstrip(';')
    if '=' not in line:
        return variables
    lhs, rhs = line.split('=', 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    value = evaluate_expression(rhs, variables)

    pattern = r'(\w+)\[(.+?)\]'
    match = re.match(pattern, lhs)
    if match:
        array_name = match.group(1)
        index_expr = match.group(2)
        index_value = evaluate_expression(index_expr, variables)
        array = variables.get(array_name)
        if array is None:
            raise Exception(f"Array '{array_name}' not found.")
        if not isinstance(index_value, int):
            raise Exception(f"Index '{index_expr}' evaluated to non-integer value.")
        if index_value < 0 or index_value >= len(array):
            raise Exception(f"Out-of-bounds write in mx.array '{array_name}' at index {index_value}.")
        array[index_value] = value
    else:
        variables[lhs] = value
    return variables

def evaluate_condition(condition, variables):
    condition = condition.replace('&&', 'and')
    condition = condition.replace('||', 'or')

    # Replace variables with their values
    for var in sorted(variables, key=lambda v: -len(v)):
        condition = re.sub(r'\b' + re.escape(var) + r'\b', str(variables[var]), condition)
    # Evaluate the condition
    try:
        return eval(condition, {"__builtins__": None}, {})
    except Exception as e:
        raise Exception(f"Error evaluating condition '{condition}': {e}")

def parse_code_blocks(code_lines):
    statements = []
    index = 0
    while index < len(code_lines):
        line = code_lines[index].strip()
        if line.startswith('if'):
            # Extract the condition
            m = re.match(r'if\s*\((.*)\)\s*{', line)
            if m:
                condition = m.group(1)
                index += 1
                # Extract the code block
                block_lines = []
                brace_count = 1
                while index < len(code_lines) and brace_count > 0:
                    line = code_lines[index]
                    if '{' in line:
                        brace_count += line.count('{')
                    if '}' in line:
                        brace_count -= line.count('}')
                    if brace_count > 0:
                        block_lines.append(line)
                    index += 1
                if brace_count != 0:
                    raise Exception("Mismatched braces in if statement")
                # Recursively parse the block
                block_statements = parse_code_blocks(block_lines)
                statements.append(('if', condition, block_statements))
            else:
                # Try to match a single-line if statement
                m = re.match(r'if\s*\((.*)\)\s*(.+);', line)
                if m:
                    condition = m.group(1)
                    single_line = m.group(2)
                    # Parse the single-line statement
                    statements.append(('if', condition, [('assign', single_line)]))
                    index += 1
                else:
                    # Handle cases where the statement is on the next line
                    m = re.match(r'if\s*\((.*)\)', line)
                    if m:
                        condition = m.group(1)
                        index += 1
                        if index < len(code_lines):
                            next_line = code_lines[index].strip()
                            statements.append(('if', condition, [('assign', next_line.rstrip(';'))]))
                            index += 1
                        else:
                            raise Exception(f"Expected statement after 'if' at line {index}")
                    else:
                        raise Exception(f"Invalid if statement syntax at line {index}: {line}")
        else:
            statements.append(('assign', line))
            index += 1
    return statements

def execute_statements(statements, variables):
    for stmt in statements:
        if stmt[0] == 'assign':
            process_assignment(stmt[1], variables)
        elif stmt[0] == 'if':
            condition = stmt[1]
            block_statements = stmt[2]
            if evaluate_condition(condition, variables):
                execute_statements(block_statements, variables)

def verify_source(source, inputs, outputs, grid):
    grid_x, grid_y, grid_z = grid
    source = preprocess_code(source)
    code_lines = [line.strip() for line in source.strip().split('\n') if line.strip()]
    statements = parse_code_blocks(code_lines)
    variables = {}
    variables.update(inputs)
    variables.update(outputs)
    errors = []

    # Simulate each thread in the grid
    for x in range(grid_x):
        for y in range(grid_y):
            for z in range(grid_z):
                thread_variables = variables.copy()
                thread_variables['thread_position_in_grid_x'] = x
                thread_variables['thread_position_in_grid_y'] = y
                thread_variables['thread_position_in_grid_z'] = z
                try:
                    execute_statements(statements, thread_variables)
                except Exception as e:
                    errors.append(e)

    if errors: 
        for e in errors: print(e)
        raise Exception(f"{len(errors)} errors found in Metal source code.")
