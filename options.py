from easydict import EasyDict

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['VIL100']
OPTION.valset = 'VIL100'
OPTION.setting = 'part_2,2_train_bk'#
OPTION.root = './dataset/VIT'  # dataset root path
OPTION.datafreq = [3] #
OPTION.max_object = 8 # max number of instances
OPTION.input_size = (240, 448)   # input image sizee
OPTION.sampled_frames =11     # min sampled time length while trianing(T)
OPTION.max_skip = [5]         # max skip time length while trianing
OPTION.samples_per_video = 1  # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 5   #T-opt.save_freq
OPTION.save_freq_max = 100
OPTION.epochs_per_increment = 2

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs =30
OPTION.train_batch = 1
OPTION.learning_rate = 0.000001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'sgd'             # 'sgd' or 'adam'
OPTION.weight_decay = 1e-7 #5e-4
OPTION.iter_size = 1
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'hope'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training

# ---------------------------------------- testing configuration --------------------------------------------
OPTION.epoch_per_test = 3

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = 'models'
OPTION.initial = ''      # path to initialize the backbone
OPTION.initial_STM =  ""  # path to initialize the backbone ./models/initial_STM.pth.tar


OPTION.resume_ATT = './models/VIL100/part_1,2_train_bk/ATT/hope90.pth.tar' # path to restart from the checkpoint
OPTION.resume_STM = './models/VIL100/part_1,2_train_bk/STM/hope90.pth.tar' # path to restart from the checkpoint recurrent35
OPTION.gpu_id = '0'      # defualt gpu-id (if not specified in cmd)
OPTION.workers = 0
OPTION.save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
OPTION.output_dir = 'output'
