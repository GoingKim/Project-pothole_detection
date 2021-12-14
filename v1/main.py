## 라이브러리 추가하기
from train import *
from util import *

import warnings
warnings.filterwarnings('ignore')

## Configurations
class CFG:
    mode = 'train'
    seed = 42
    print_freq = 30

    n_class = 1

    img_size = 512

    num_fold = 5
    num_epoch = 40
    batch_size = 4
    num_workers = 4    # decide how many data upload to dataset for a batch
                       # if n_workers 2, dataloader works twice for a batch.
                       # It has impact for cuda memory too

    lr = 0.0002

    max_grad_norm = 1000

    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )

    # data_dir = './data_pothole'
    ckpt_dir = './checkpoint'
    result_dir = './result'
    log_dir = './log'

    csv_dir = './train_df.csv'
    csv2_dir = './train_df_kfold.csv'


##
if __name__ == "__main__":
    seed_everything(CFG.seed)
    df = pd.read_csv(CFG.csv_dir)
    df_kfold = pd.read_csv(CFG.csv2_dir)

    if CFG.mode == 'train':
        torch.set_default_tensor_type('torch.FloatTensor')
        fit(df, df_kfold)