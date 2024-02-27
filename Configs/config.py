import os
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 1337
C.repo_name = "medical_semi_seg"


""" Experiments setting """
C.augmentation = True
C.dataset = "Pancreas"
C.code_path = os.path.realpath("") #+ "/Code/VnetPancreas/"
C.data_path = os.path.realpath("Datasets/Pancreas/data/".format("."))

""" Training setting """
# trainer
C.ddp_training = False
C.batch_size = 4
C.num_workers = 1
C.learning_rate = 2.5e-2
C.shuffle = True
C.drop_last = False
C.threshold = 0.65
C.spatial_weight = 1.0
C.hyp = .1

# rampup settings (per epoch)
C.rampup_type = "sigmoid"
C.rampup_length = 40
C.rampup_start = 0

""" Model setting """
C.num_classes = 2
C.momentum = 0.9
C.weight_decay = 1e-4
C.ema_momentum = 0.99

""" Wandb setting """
os.environ["WANDB_API_KEY"] = "7fb4766980dd9063b4834ff7fac76a6849ca1aa1"
C.use_wandb = False
C.project_name = "VNET"
C.pvc = False

""" Others """
C.save_ckpt = True
C.pvc = False


C.last_val_epochs = 0
