import os
import warnings
import time
import torch
import wandb

from utils.utils import config2string
from utils.argument import parse_args
from utils.utils import train_reports


def main(args, trial):
    if args.wandb:
        if args.wb_name is not None:
            wandb.init(project=args.project, name=args.wb_name,
                       settings={'init_timeout': 600})
        else:
            wandb.init(project=args.project, name=args.embedder,
                       settings={'init_timeout': 600})
        wandb.config.update(args)

      
    if args.embedder.lower() == 'spotscape':
        from models.Spotscape import Spotscape_Trainer as Trainer
    elif args.embedder.lower() == 'spotscape_large':
        from models.Spotscape_Large import Spotscape_Trainer as Trainer


    modeler = Trainer(args, trial)
    res = modeler.train()
    
    if args.wandb:
        wandb.finish()
    if args.report:
        torch.save(modeler.model.state_dict(),
                   os.path.join(args.result_dir, args.embedder, f'{args.config_str}_trial_{trial}.pt'))

    return modeler, res


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    torch.set_num_threads(4)
    args, unknown = parse_args()

    args.config_str = config2string(args)
    print("\n[Config] {}\n".format(args.config_str))

    ############################ For Recording ############################
    args.result_path = os.path.join(args.result_dir, args.embedder,
                                    f'{args.dataset}_{args.patient_idx}.txt')
    if args.result_dir in ['./reports', './reports/', './report', './report/']:
        args.report = True
        args.timestr = time.strftime('%m%d')
    else:
        args.report = False
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)

    train_reports(args, main)

