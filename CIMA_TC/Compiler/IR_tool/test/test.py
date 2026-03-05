from ..core import *
from ..ops import *
if __name__ == "__main__":


    node = make_layer({
        'type': 'op',
        'op': 'relu',
        'inputs': {'x': DataDef()},
        'outputs': {'y': DataDef()}
    })

    print(type(node.inputs['x'].ref))