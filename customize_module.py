import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from time_series_models import TimeSeriesModel, Attention

class FlexibleModelCompiler:
    """
    Compiles a PyTorch TimeSeriesModel subclass based on a configuration dict.

    Config schema:
      class_name: str
      base_class: nn.Module subclass
      init_params: dict of constructor parameters and their default values
      layers: dict mapping attribute names to (factory_fn_name: str, *args)
              where factory_fn_name can be 'gru', 'lstm', 'linear', 'dropout'
      forward: dict mapping var names to Python expressions (strings)

    Generates a class that:
      - Inherits from base_class
      - Defines __init__ with parameters and defaults
      - Calls super().__init__(<init_param_names>)
      - Instantiates layers with auto kwargs for recurrent layers
      - Defines forward(self, x) based on forward config, translating '+' to torch.cat and layer calls to [0] captures
    """
    def __init__(self, config: dict):
        self.cfg = config
        self._validate()
        # store layer keys for use in replacement
        self.layer_names = list(self.cfg['layers'].keys())

    def _validate(self):
        required = ['class_name', 'base_class', 'init_params', 'layers', 'forward']
        for key in required:
            if key not in self.cfg:
                raise ValueError(f"Missing config key: {key}")
        if not isinstance(self.cfg['init_params'], dict):
            raise ValueError("'init_params' must be a dict of name->default values")
        if not isinstance(self.cfg['layers'], dict) or not isinstance(self.cfg['forward'], dict):
            raise ValueError("'layers' and 'forward' must be dicts")

    def replace_shortened_functions(self, expr: str) -> str:
        """
        Replace shorthand layer calls with self-prefixed calls
        based on configured layer names.
        """
        for name in self.layer_names:
            # replace e.g. 'fc(' with 'self.fc('
            expr = re.sub(rf"\b{name}\(", f"self.{name}(", expr)
        return expr

    def generate_model(self):
        class_name = self.cfg['class_name']
        base = self.cfg['base_class']
        init_params = self.cfg['init_params']
        layers_cfg = self.cfg['layers']
        forward_cfg = self.cfg['forward']

        lines = []
        # Class header
        lines.append(f"class {class_name}({base}):")
        # __init__ signature
        params_sig = ', '.join(f"{k}={repr(v)}" for k, v in init_params.items())
        lines.append(f"    def __init__(self, {params_sig}):")
        # super init
        if base == 'TimeSeriesModel':
            names = ', '.join(init_params.keys())
            lines.append(f"        super().__init__({names})")
        else:
            lines.append(f"        super().__init__()")
        
        # Instantiate layers
        for attr, spec in layers_cfg.items():
            fn = spec[0].lower()
            args = spec[1:]
            if fn in ('gru', 'lstm'):
                inp, hid, nl, dp = args
                cls = 'nn.GRU' if fn == 'gru' else 'nn.LSTM'
                lines.append(
                    f"        self.{attr} = {cls}("
                    f"{inp}, {hid}, {nl}, bias=True, batch_first=True, dropout={dp})"
                )
            elif fn == 'linear':
                inp, outp = args
                lines.append(f"        self.{attr} = nn.Linear({inp}, {outp})")
            elif fn == 'dropout':
                p, = args
                lines.append(f"        self.{attr} = nn.Dropout({p})")
            elif fn == 'attention':
                p, = args
                lines.append(f"        self.{attr} = Attention({p})")
            # Activation: ReLU
            elif fn == 'relu':
                lines.append(f"        self.{attr} = nn.ReLU()")
            # Activation: Tanh
            elif fn == 'tanh':
                lines.append(f"        self.{attr} = nn.Tanh()")
            # Sequential container
            elif fn in ('sequential', 'nn.sequential'):
                # args are module definition strings
                lines.append(f"        self.{attr} = nn.Sequential(")
                for module_str in args:
                    lines.append(f"            {module_str},")
                # remove trailing comma from last
                lines[-1] = lines[-1].rstrip(',')
                lines.append(f"        )")
            else:
                args_list = ', '.join(map(str, args))
                lines.append(f"        self.{attr} = {fn}({args_list})")

        # forward method
        lines.append("")
        lines.append("    def forward(self, x: torch.Tensor):")
        # Translate forward config to code lines

        for var, expr in forward_cfg.items():
            # handle concatenation via '+'
            if '+' in expr and 'torch.' not in expr and '(' not in expr:
                parts = [p.strip() for p in expr.split('+')]
                concat = ', '.join(parts)
                code = f"        {var} = torch.cat([{concat}], dim=-1)"
            # handle layer calls e.g. 'fc(x), dropout(x), fc(dropout(x))'
            else:
                # apply shorthand replacement
                expr = self.replace_shortened_functions(expr)
                code = f"        {var} = {expr}"
            lines.append(code)
        lines.append("        return output")

        # Execute dynamic code\        
        code_str = '\n'.join(lines)
        self.class_code = code_str
        # exec_globals = {
        #     're': re,
        #     'torch': torch,
        #     'nn': nn,
        #     'F': F,
        #     base.__name__: base,
        #     'Attention': Attention
        # }
        # namespace = {}
        # exec(code_str, exec_globals, namespace)

        namespace = globals()
        exec(code_str, namespace)
        return namespace[class_name]