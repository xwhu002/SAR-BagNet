import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        print('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
