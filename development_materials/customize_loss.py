import ast
import torch
import torch.nn as nn
from collections import OrderedDict
import re

import numpy as np
import pandas as pd

# Must define this function here, otherwise the namespace = globals() can't find this function
def Z_norm_reverse(X,Xscaler,units_convert=1.0):
    return (X*Xscaler[1]+Xscaler[0])*units_convert

def safe_repr(value):
    """
    Returns a string representation of 'value'.
    - For basic Python types (int, float, str, list, tuple, set), uses repr() directly.
    - For NumPy arrays, Pandas DataFrames, and Torch Tensors,
      converts them to a Python list before using repr().
    """
    if isinstance(value, (int, float, str, list, tuple, set, dict, bool, type(None))):
        return repr(value)
    elif isinstance(value, np.ndarray):
        return repr(value.tolist())
    elif isinstance(value, pd.DataFrame):
        # Convert DataFrame to a list of lists (rows) or dict of lists (columns)
        # Choosing rows here for a common list-like representation
        return repr(value.values.tolist())
    elif isinstance(value, torch.Tensor):
        return repr(value.tolist())
    else:
        # Fallback for other types
        return repr(value)

class CarbonFluxLossCompiler:
    def __init__(self, script_config):
        self.script_config = script_config
        self.validate_config()
        self.analyze_dependencies()
        self.generate_class_code()
        
    def validate_config(self):
        """Validate configuration format"""
        required_sections = ['parameters', 'variables', 'loss_fomula']
        for section in required_sections:
            if section not in self.script_config:
                raise ValueError(f"Script config must contain '{section}' section")
                
        # Ensure the loss formula contains the final loss expression
        if 'loss' not in self.script_config['loss_fomula']:
            raise ValueError("Loss formula must contain a 'loss' expression")
    
    def extract_all_expressions(self):
        """Extracts all expressions into a dictionary"""
        expressions = {}
        
        # Add parameters
        expressions.update(self.script_config['parameters'])
        
        # Add variable expressions
        expressions.update(self.script_config['variables'])
        
        # Add loss formulas
        expressions.update(self.script_config['loss_fomula'])
        
        return expressions
    
    def analyze_dependencies(self):
        """Analyzes dependencies between expressions"""
        # Extract all expressions
        all_expressions = self.extract_all_expressions()
        
        self.dependencies = OrderedDict()
        self.tensor_extractions = []
        self.intermediate_exprs = []
        
        # Define list of known function names (these should not be considered variable dependencies)
        self.known_functions = {'mean', 'abs', 'sum', 'exp', 'log', 'sqrt', 'min', 'max', 'Z_norm_reverse', 'relu'}
        
        # 1. Create dependency graph
        for key, expr in all_expressions.items():
            if not isinstance(expr, str):
                continue
                
            try:
                tree = ast.parse(expr, mode='eval')
            except SyntaxError as e:
                raise ValueError(f"Invalid expression for '{key}': {e}")
                
            variables = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id != 'self' and node.id not in self.known_functions:
                    variables.add(node.id)
            
            self.dependencies[key] = variables
        
        # 2. Identify tensor extraction expressions (those that depend only on inputs or class attributes)
        for key, deps in self.dependencies.items():
            if key == 'loss':
                continue  # Skip final loss
                
            # Check if it only depends on input tensors or configuration parameters
            if all(dep in ['batch_x', 'y_pred', 'y_true'] or 
                   (dep in all_expressions and not isinstance(all_expressions[dep], str))
                   for dep in deps):
                self.tensor_extractions.append(key)
        
        # 3. Identify intermediate expressions (dependent on other expressions)
        self.intermediate_exprs = [key for key in self.dependencies.keys() 
                                  if key != 'loss' and key not in self.tensor_extractions]
        
        # 4. Determine evaluation order for intermediate expressions
        evaluated = set(self.tensor_extractions)
        available_vars = set(self.tensor_extractions) | set(
            k for k, v in all_expressions.items() if not isinstance(v, str)
        )
        
        # Topological sort
        evaluation_order = []
        while self.intermediate_exprs:
            added = False
            remaining = []
            
            for key in self.intermediate_exprs:
                deps = self.dependencies[key]
                
                # Check if all dependencies are available
                if deps.issubset(available_vars):
                    evaluation_order.append(key)
                    available_vars.add(key)
                    added = True
                else:
                    remaining.append(key)
            
            if not added and remaining:
                # Find missing dependencies
                missing_deps = {}
                for key in remaining:
                    deps = self.dependencies[key]
                    missing = deps - available_vars
                    if missing:
                        missing_deps[key] = missing
                
                error_msg = "Circular dependency detected or missing variables:\n"
                for key, missing in missing_deps.items():
                    error_msg += f"  - '{key}' missing: {', '.join(missing)}\n"
                raise RuntimeError(error_msg)
            
            self.intermediate_exprs = remaining
        
        self.evaluation_order = evaluation_order
    
    def replace_shortened_functions(self, expr):
        """Replace abbreviated function names with their full torch-prefixed forms"""
        replacements = {
            r'\bmean\(': 'torch.mean(',
            r'\babs\(': 'torch.abs(',
            r'\bsum\(': 'torch.sum(',
            r'\bexp\(': 'torch.exp(',
            r'\blog\(': 'torch.log(',
            r'\bsqrt\(': 'torch.sqrt(',
            r'\bmin\(': 'torch.min(',
            r'\bmax\(': 'torch.max(',
            r'\brelu\(': 'torch.relu(',
            # r'\bZ_norm_reverse\(': 'self.Z_norm_reverse(',
        }
        
        for pattern, replacement in replacements.items():
            expr = re.sub(pattern, replacement, expr)
        
        return expr
    
    def generate_class_code(self):
        """Generate Python source code for the CarbonFluxLoss class"""
        # Get parameter configuration
        parameters = self.script_config['parameters']
        all_expressions = self.extract_all_expressions()
        
        # Class header
        class_code = [
            "import torch",
            "import torch.nn as nn",
            "",
            "class CarbonFluxLoss(nn.Module):",
        ]
        
        # Generate __init__ method parameter list
        init_params = []
        for key, value in parameters.items():
            # For scaler parameters, use default value None
            # if key in ['GPP_scaler', 'y_scaler']:
            #     init_params.append(f"{key}=None")
            # else:
            #     init_params.append(f"{key}={repr(value)}")

            init_params.append(f"{key}={safe_repr(value)}")
                
        class_code.append(f"    def __init__(self, {', '.join(init_params)}):")
        class_code.append("        super().__init__()")
        
        # Store all parameters as class attributes
        for key in parameters.keys():
            # For scalers, store them as tensors that do not require gradients
            # if key in ['GPP_scaler', 'y_scaler']:
            #     class_code.append(f"        self.{key} = torch.tensor({key}, requires_grad=False) if {key} is not None else None")
            # else:
            #     class_code.append(f"        self.{key} = {key}")

            class_code.append(f"        self.{key} = {key}")
        
        # Add Z_norm_reverse method
        # class_code.extend([
        #     "",
        #     "    def Z_norm_reverse(self, x, scaler):",
        #     "        \"\"\"Z norm reverse\"\"\"",
        #     "        if scaler is None:",
        #     "            return x",
        #     "        return x * scaler[1] + scaler[0]"
        # ])
        
        # Process other non-parameter attributes (if any)
        for key, value in all_expressions.items():
            if key in parameters:  # Skip processed parameters
                continue
                
            if not isinstance(value, str):
                if isinstance(value, (int, float)):
                    class_code.append(f"        self.{key} = {value}")
                # If the parameter is a tensor, store it as a tensor that does not require gradients
                elif torch.is_tensor(value):
                    class_code.append(f"        self.{key} = torch.tensor({value.tolist()}, requires_grad=False)")
                # Other types are stored directly
                else:
                    class_code.append(f"        self.{key} = {repr(value)}")
        
        # Forward method header
        class_code.extend([
            "",
            "    def forward(self, y_pred, y_true, batch_x):"
        ])
        
        # Copy parameters to local variables
        param_names = list(parameters.keys())
        if param_names:
            class_code.append("        # Copy parameters to local variables")
            for name in param_names:
                class_code.append(f"        {name} = self.{name}")
        
        # Tensor extraction
        if self.tensor_extractions:
            class_code.append("\n        # Tensor extraction")
            for key in self.tensor_extractions:
                expr = all_expressions[key]
                # Replace shorthand function names with full torch-prefixed forms
                expr = self.replace_shortened_functions(expr)
                class_code.append(f"        {key} = {expr}")
        
        # Intermediate calculations
        if self.evaluation_order:
            class_code.append("\n        # Intermediate calculations")
            for key in self.evaluation_order:
                expr = all_expressions[key]
                # Replace shorthand function names with full torch-prefixed forms
                expr = self.replace_shortened_functions(expr)
                class_code.append(f"        {key} = {expr}")
        
        # Add loss
        class_code.append("\n        # Loss")
        loss_expr = self.replace_shortened_functions(all_expressions['loss'])
        class_code.append(f"        loss = {loss_expr}")
        class_code.append("        return loss")
        
        self.class_code = "\n".join(class_code)
    
    def compile_class(self):
        """Compile and return the CarbonFluxLoss class"""
        # namespace = {
        #     'torch': torch,
        #     'nn': nn,
        #     '__name__': '__carbon_flux_loss__'
        # }
        namespace = globals()
        
        try:
            exec(self.class_code, namespace)
        except Exception as e:
            print("Generated code:")
            print(self.class_code)
            raise RuntimeError(f"Failed to compile CarbonFluxLoss class: {e}")
        
        return namespace['CarbonFluxLoss']
    
    def generate_class(self):
        return self.compile_class()

def generate_carbon_flux_loss_class(script_config):
    compiler = CarbonFluxLossCompiler(script_config)
    return compiler.generate_class()