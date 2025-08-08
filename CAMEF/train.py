from data.dataloader import event_set
import os
from model.CAMEF2 import CAMEF, train, test
import torch
import warnings

# 屏蔽所有警告
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":


    seq_len=35
    pred_len=35
    event_id  = 0
    series_id = 'SP500'
    # moment_model = os.path.join('model','moment')
    # bert_model = os.path.join('model','longformer')
    # gpt_model = os.path.join('model','gpt2')
    gpt_model = "openai-community/gpt2"
    # bert_model = "allenai/longformer-base-4096"
    bert_model = "FacebookAI/roberta-base"
    # moment_model = "AutonLab/MOMENT-1-large"
    moment_model = "/PATH/TO/FINETUNED/MOMENT/MODEL"

    batch_size = 10

    window=500
    stride=400
    event_set = event_set(
        seq_len, pred_len,

        event_id=event_id,
        series_id= series_id,

        shuffle = True,
        batch_size = batch_size,

        scale=True,
        event_dir='/PATH/TO/EVENT/FOLDER',
        series_dir='/PATH/TO/TIME-SERIES/FOLDER',

    )

    train_loader, test_loader, vali_loader = event_set.train_loader, event_set.test_loader, event_set.vali_loader
    model = CAMEF(
        bert= bert_model,
        moment= moment_model,
        gpt= gpt_model,

        seq_len=seq_len,
        pred_len=pred_len,
        d = event_set.d,

        window=window,
        stride=stride,

        batch_size=batch_size,
    )


    output_path = "/OUTPUT/PATH"

    train(model, train_loader, test_loader, vali_loader, num_epochs=10, output_path=output_path)
