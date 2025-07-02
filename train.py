import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from dataset.mydata import MyData
from models.mymodel import MyModel
from trainer.training import train_loop

# CUDA environment
N_GPU = 1
device_ids = [i for i in range(N_GPU)]

root_path = 'path/to/data'
train_video_dir = 'path/train'
train_label_dir = 'data_train.csv'
valid_video_dir = 'path/val'
valid_label_dir = 'data_eval.csv'     

NUM_FRAME = 64
IMG_SIZE = 112
NUM_EPOCHS = 600
LR = 5e-6
BATCH_SIZE = 4

train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME, img_size=IMG_SIZE)
valid_dataset = MyData(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME, img_size=IMG_SIZE)
my_model = MyModel(NUM_FRAME)
print(len(train_dataset), len(valid_dataset))

train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset,
           batch_size=BATCH_SIZE, lr=LR, ckpt_name='', lastckpt=None, 
           log_dir='', device_ids=device_ids)
#valid(my_model, valid_dataset, batch_size=1, lastckpt=None, device_ids=device_ids)