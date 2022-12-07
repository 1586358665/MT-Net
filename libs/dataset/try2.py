

from libs.dataset.transform import TrainTransform
from libs.dataset.data import *

from PIL import Image
class get_data(BaseData):
    def __init__(self,transform=None):
        # input_dim = opt.input_size
        # train_transformer = TrainTransform(size=input_dim)
        # transform = train_transformer
        self.transform = transform

    def start(self,frame_path,mask_path):

        frame = [np.array(Image.open(frame_path[i])) for i in range(6)]
        mask = [np.array(Image.open(mask_path[i])) for i in range(6)]

        mask = [convert_mask(msk, 8) for msk in mask]


        frame, mask = self.transform(frame, mask, False)

        return frame, mask
    def batch( self,frame, masks, num_objects):


        # memorize a frame
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        bg_batch = []

        for o in range(1, num_objects + 1):  # 1 - no
            frame_batch.append(frame)
            mask_batch.append(masks[:, o])

        for o in range(1, num_objects + 1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0)) #背景mask

        # make Batch
        in_f = torch.cat(frame_batch, dim=0)
        in_m = torch.cat(mask_batch, dim=0)
        in_bg = torch.cat(bg_batch, dim=0)
        frame_batch= in_f
        mask_batch = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        bg_batch = torch.unsqueeze(in_bg, dim=1).float()

        return  frame_batch,mask_batch,bg_batch




