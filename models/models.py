import copy
import torch

models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# Registration function to associate model names with classes/functions
def register_model(name, model_class):
    models[name] = model_class


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    # except:
    #     print(model_spec)
    if load_sd:
        model.load_state_dict(model_spec['sd'], strict=False)
    return model


def compare_models(model1, model2):
    for k,v in model1.items():
        if torch.equal(v, model2[k].to(v.device)):
            continue
        else:
            return False
    return True

