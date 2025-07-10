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
        lines.append("")

        # Save input parameters
        for key in init_params.keys():
            lines.append(f"        self.{key} = {key}")
        lines.append("")

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

        # copy parameters to local var
        lines.append("        # Copy parameter to local")
        for key in init_params.keys():
            lines.append(f"        {key} = self.{key}")
        lines.append("")

        concat_symbol = '&'
        mm_symbol = '@'
        for var, expr in forward_cfg.items():
            # handle concatenation
            if concat_symbol in expr and 'torch.' not in expr and '(' not in expr:
                parts = [p.strip() for p in expr.split(concat_symbol)]
                concat = ', '.join(parts)
                code = f"        {var} = torch.cat([{concat}], dim=-1)"
            # handle layer calls e.g. 'fc(x), dropout(x), fc(dropout(x))'
            elif mm_symbol in expr :
                parts = [p.strip() for p in expr.split(mm_symbol)]
                matrix_multiple = ', '.join(parts)
                code = f"        {var} = torch.bmm({matrix_multiple})"
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

# Example usage:
if __name__ == '__main__':

    config_kgml = {
        'class_name': 'my_KGML',
        'base_class': 'TimeSeriesModel',
        'init_params': {
            'input_dim': 19, # number of input features
            'hidden_dim': 128,
            'num_layers': 2,
            'output_dim': 3,
            'dropout': 0.2
        },
        'layers': {
            'gru_basic': ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout'), # basic
            'gru_ra':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', 'num_layers', 'dropout'), # Ra prediction
            'gru_rh':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', '1', '0'),
            'gru_nee':   ('gru', 'input_dim+2', 'hidden_dim', '1', '0'), #
            'dropout':   ('dropout', 'dropout'),
            'fc':        ('linear', 'hidden_dim', '1'),
            'attn1':      ('Attention', 'hidden_dim'),
            'attn2': ('Sequential', 'nn.Linear(hidden_dim, 64)', 'nn.ReLU()', 'nn.Linear(64, 64)', 'nn.ReLU()', 
                      'nn.Linear(64, 32)', 'nn.ReLU()', 'nn.Linear(32, 1)', 'nn.Tanh()'),
            'attn3': ('my_attention', 'hidden_dim'),
            'attn4': ('dual_attention', 'hidden_dim')


        },
        'forward': {
            'out_basic, hidden': 'gru_basic(x)',
            'dropped':   'dropout(out_basic)',
            'attend':    'attn4(dropped)',
            # attend and input together
            'ra_in':     'attend & x',
            'ra_out, hidden':    'gru_ra(ra_in)',  # Ignor hidden
            'ra_pred':   'fc(dropout(ra_out))',
            'rh_in':     'attend & x',
            'rh_out, hidden':    'gru_rh(rh_in)',
            'rh_pred':   'fc(dropout(rh_out))',
            'nee_in':    'ra_pred & rh_pred & x',
            'nee_out, hidden':   'gru_nee(nee_in)',
            'nee_pred':  'fc(dropout(nee_out))',
            'output':    'ra_pred & rh_pred & nee_pred'
        }
    }

    config_attn = {
        'class_name': 'my_attention',
        'base_class': 'nn.Module',
        'init_params': {
            'hidden_dim': 128,
        },
        'layers': {
            'fc1':        ('linear', 'hidden_dim', '64'),
            'relu': ('relu',),
            'fc2': ('linear', '64', '64'),
            'fc3': ('linear', '64', '32'),
            'fc4': ('linear', '32', '1'),
            'tanh': ('tanh',)
        },
        'forward': {
            'x1': 'fc1(x)',
            'x2':   'relu(x1)',
            'x3':    'fc2(x2)',
            'x4':     'relu(x3)',
            'x5':    'fc3(x4)',
            'x6':   'relu(x5)',
            'x7':    'fc4(x6)',
            'x8':   'tanh(x7)',
            'output':    'x8'
        }
    }

    dual_attn = {
        'class_name': 'dual_attention',
        'base_class': 'nn.Module',
        'init_params': {
            'hidden_dim': 128,
        },
        'layers':{
            'fc':        ('linear', 'hidden_dim', 'hidden_dim'),

        },
        'forward': {
            'Q': 'fc(x)',
            'K': 'fc(x)',
            'V': 'fc(x)',
            '_scores': 'Q @ K.transpose(1, 2)',  #torch.bmm(Q, K.transpose(1, 2)) / self.scale
            'scores': '_scores/ (hidden_dim**0.5)',
            'attention_weights': 'F.softmax(scores, dim=-1)',
            'context': 'attention_weights @ V', #torch.bmm(attention_weights, V)
            'output': 'x & context',
        }
    }

    Compiler1 = FlexibleModelCompiler(config_attn)
    Compiler1.generate_model()

    print("\n Generated class code:")
    print(Compiler1.class_code)

    Compiler2 = FlexibleModelCompiler(dual_attn)
    my_dual_attn = Compiler2.generate_model()
    print("\n Generated class code:")
    print(Compiler2.class_code)

    Compiler = FlexibleModelCompiler(config_kgml)
    myKGML = Compiler.generate_model()
    print("\n Generated class code:")
    print(Compiler.class_code)

    model = myKGML(19, 128, 3, 2, 0.2)
    inp = torch.randn(4, 50, 19)
    print(model(inp).shape)  # -> (4, 50, 3)
