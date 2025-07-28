import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from time_series_models import TimeSeriesModel, Attention

def extract_functions(expr: str) -> tuple[list[str], list[str]]:
    """
    Given an expression like 'fc(dropout(rh_out))' or 'attend & x',
    return a list of function names and their direct arguments.
    - For 'fc(dropout(rh_out))' returns ['fc', 'dropout'], ['rh_out']
    - For 'nn.Linear(64,32)' returns ['nn.Linear'], ['64', '32']
    - For 'attend & x' (no function) returns ['attend & x']
    """
    # 1) Find all function names (identifier followed by '(')
    func_names = re.findall(r'\b(\w+(?:\.\w+)*)\s*\(', expr)

    # 2) Capture the innermost parenthesis content (no nested '(' or ')')
    inner_args = re.findall(r'\(\s*([^()]+?)\s*\)', expr)
    # Filter out any that still contain parentheses
    args = [arg for arg in inner_args if '(' not in arg and ')' not in arg]
    # args will like ['64, 32']
    parts = [p.strip() for p in args[0].split(',')] if args else []

    # 3) If we found any functions, return them plus the direct args
    if func_names:
        return func_names, parts

    # 4) Otherwise there’s no function call—return the raw expression
    return [], [expr.strip()]


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

    def check_configuration(self):
        init_params = self.cfg['init_params']
        layers_cfg = self.cfg['layers']
        forward_cfg = self.cfg['forward']
        
        # Instantiate layers
        layers_dict = {}
        layers_name = list()
        fn_list = list() # record all functions in layers
        layers_keys = list(layers_cfg.keys())
        for attr, spec in layers_cfg.items():
            fn = spec[0].strip().lower()
            fn_list.append(fn)
            layers_name.append(attr.strip())

            args = spec[1:]
            if fn in ('gru', 'lstm'):
                inp, hid, *rest = args
                cls = 'nn.GRU' if fn == 'gru' else 'nn.LSTM'
                outp = hid
                layers_dict[attr] = [cls, inp, outp]
            elif fn == 'linear':
                inp, outp = args
                layers_dict[attr] = ['nn.Linear', inp, outp]
            elif fn == 'dropout':
                p, = args
                layers_dict[attr] = ['nn.Dropout', 0, 0]
            elif fn == 'attention':
                p, outp = args
                layers_dict[attr] = ['Attention', p, outp]
            # Activation: ReLU
            elif fn == 'relu':
                layers_dict[attr] = ['nn.ReLU', 0, 0]
            # Activation: Tanh
            elif fn == 'tanh':
                layers_dict[attr] = ['nn.Tanh', 0, 0]
            elif fn == 'softmax' or fn == 'F.softmax':
                layers_dict[attr] = ['F.softmax', 0, 0]
            # Sequential container
            elif fn in ('sequential', 'nn.sequential'):
                for func_call in args:
                    fn,params = extract_functions(func_call)
                    if fn == ['nn.Linear']:
                        inp, outp = params
                layers_dict[attr] = ['nn.Sequential', 0, outp] # get last nn.Linear()
            else:
                inp, outp = args
                layers_dict[attr] = [fn, inp, outp]

        fn_set = set(fn_list) #remove duplicate items

        # Forward process
        concat_symbol = '&'
        mm_symbol = '@'
        dot_symbol = '.'

        return_p_list = ['x']
        output_p_dict = dict()
        init_params_keys = list(init_params.keys())
        output_p_dict['x'] = init_params_keys[0] # Get first key

        for var, expr in forward_cfg.items():
            # get return parameters
            return_p = [part.strip() for part in var.split(',')]
            return_p_list += return_p

            if concat_symbol in expr and 'torch.' not in expr and '(' not in expr:
                parts = [p.strip() for p in expr.split(concat_symbol)]
                # each part should exist in previous
                for part in parts:
                    if part not in return_p_list:
                        print(f"Warning check {part} in the {expr}")

                total_dim = []
                for part in parts:
                    _dim = output_p_dict[part]
                    total_dim.append(_dim)

                full_dim = '+'.join(total_dim)
                for p in return_p:
                    output_p_dict[p] = full_dim

            elif mm_symbol in expr :
                parts = [p.strip() for p in expr.split(mm_symbol)]
                # Get last matrix's dimmension
                last_part = parts[-1]
                for p in return_p:
                    if last_part in return_p_list:
                        output_p_dict[p] = output_p_dict[last_part]
                    else:
                        output_p_dict[p] = -1 # means unknow dimmension

            elif dot_symbol in expr: # exist like 0.5 F.softmax(), torch.sqrt(), nn.zeros()
                skip_line = False
                fns,params = extract_functions(expr) # get [fns], [params]
                if len(fns) == 0: # no function exist
                    skip_line = True
                else:
                    for fn in fns:
                        if dot_symbol in fn:
                            skip_line = True
                
                # do nothing, skip this line
                if skip_line:
                    continue

            else:
                # extract func names and parameters
                fns,params = extract_functions(expr) # get [fns], [params]
                fns.reverse() # change func order for case: ['fc', 'dropout'], change to ['dropout', 'fc']
                if len(fns) == 0: # no function in configuration the the fns is a empty list
                    output_p_dict[var] = expr
                    continue

                # Input parameter should exist in before rows
                for param in params: # Only support one param now
                    if param not in return_p_list: # The param should be pre lines output
                        print(f"Warning check {param} in the {expr}")
                
                # the function name should exist in layers
                for fn in fns: 
                    if fn not in layers_keys:
                        print(f"Invalid function name {fn} in the {expr}")

                # get each func call returned dimension
                for fn in fns:
                    input_dim, output_dim = layers_dict[fn][1:]
                    if output_dim == 0: 
                        # don't change dimmension, need get original dim from input_dim
                        output_dim = output_p_dict[params[0]]

                    for p in return_p:
                        output_p_dict[p] = output_dim

                for param in params:
                    if param == 'x':
                        continue

                    if input_dim == 0:
                        continue

                    try:
                        input_para_dim_value = eval(output_p_dict[param], {}, init_params)
                        layer_required_dim = eval(input_dim, {}, init_params)
                    except NameError as e:
                        print(f"Caught a ValueError: {e}")
                        print(f"Warning check {param} in the {expr}")

                    if input_para_dim_value != layer_required_dim:
                        print(f"Warning check {param} in the {expr}")


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
                inp, hid, nl, dp, outp = args
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
                p, outp = args
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
                for module_str in args[:-1]:
                    lines.append(f"            {module_str},")
                # remove trailing comma from last
                lines[-1] = lines[-1].rstrip(',')
                lines.append(f"        )")
            else:
                args_list = ', '.join(map(str, args[:-1]))
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
    config_sample = {
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
                'gru_basic': ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout', 'hidden_dim'), # basic
                'gru_ra':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', 'num_layers', 'dropout', 'hidden_dim'), # Ra prediction
                'gru_rh':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', '1', '0', 'hidden_dim'),
                'gru_nee':   ('gru', 'input_dim+2', 'hidden_dim', '1', '0', 'hidden_dim'), #

            },
            'forward': {
                'out_basic, hidden': 'gru_basic(x)',
                'dropped':   'dropout(out_basic)',
                'attend':    'attn4(dropped)',
                # attend and input together
                'ra_in':     'attend & x',
                'ra_out, hidden':    'gru_ra(ra_in)',  # Ignor hidden
                'output':    'ra_pred & rh_pred & nee_pred'
            }
        }


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
            'gru_basic': ('gru', 'input_dim', 'hidden_dim', 'num_layers', 'dropout', 'hidden_dim'), # basic
            'gru_ra':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', 'num_layers', 'dropout', 'hidden_dim'), # Ra prediction
            'gru_rh':    ('gru', 'input_dim + 2*hidden_dim', 'hidden_dim', '1', '0', 'hidden_dim'),
            'gru_nee':   ('gru', 'input_dim+2', 'hidden_dim', '1', '0', 'hidden_dim'), #
            'dropout':   ('dropout', 'dropout'),
            'fc':        ('linear', 'hidden_dim', '1'),
            'attn1':      ('Attention', 'hidden_dim', 'hidden_dim*2'),
            'attn2': ('Sequential', 'nn.Linear(hidden_dim, 64)', 'nn.ReLU()', 'nn.Linear(64, 64)', 'nn.ReLU()', 
                      'nn.Linear(64, 32)', 'nn.ReLU()', 'nn.Linear(32, 1)', 'nn.Tanh()', '1'),
            'attn3': ('my_attention', 'hidden_dim', '1'),
            'attn4': ('dual_attention', 'hidden_dim', 'hidden_dim + hidden_dim')
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

    config_kgml_2 = {
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
            'attn1':      ('Attention', 'hidden_dim', 'hidden_dim*2'),
            'attn2': ('Sequential', 'nn.Linear(hidden_dim, 64)', 'nn.ReLU()', 'nn.Linear(64, 64)', 'nn.ReLU()', 
                      'nn.Linear(64, 32)', 'nn.ReLU()', 'nn.Linear(32, 1)', 'nn.Tanh()'),
            'attn3': ('my_attention', 'hidden_dim', '1'),
            'attn4': ('dual_attention', 'hidden_dim', 'hidden_dim + hidden_dim')
        },
        'forward': {
            'out_basic, hidden': 'gru_basic(x)',
            'dropped':   'dropout(out_basic)',
            'attend':    'attn4(dropped)',
            # attend and input together
            'ra_in':     'attend & x',
            'ra_out, hidden':    'gru_ra(ra_in)',  # Ignor hidden
            'ra_pred':   'fc(dropout(ra_out))',
            'ra_pred_attn': 'attn2(dropout(ra_out))',
            'rh_in':     'attend & x',
            'rh_out, hidden':    'gru_rh(rh_in)',
            'rh_pred':   'fc(dropout(rh_out))',
            'nee_in':    'ra_pred_attn & ra_pred & rh_pred & x',
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

    
    print("\n")
    
    Compiler1 = FlexibleModelCompiler(config_kgml_2)
    Compiler1.check_configuration()

    Compiler1 = FlexibleModelCompiler(config_attn)
    Compiler1.check_configuration()
    Compiler1.generate_model()

    print("\n Generated class code:")
    print(Compiler1.class_code)

    Compiler2 = FlexibleModelCompiler(dual_attn)
    Compiler2.check_configuration()
    my_dual_attn = Compiler2.generate_model()
    print("\n Generated class code:")
    print(Compiler2.class_code)

    Compiler = FlexibleModelCompiler(config_kgml)
    Compiler.check_configuration()
    myKGML = Compiler.generate_model()
    print("\n Generated class code:")
    print(Compiler.class_code)

    model = myKGML(19, 128, 3, 2, 0.2)
    inp = torch.randn(4, 50, 19)
    print(model(inp).shape)  # -> (4, 50, 3)
