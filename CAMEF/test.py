import argparse
import warnings

import torch

from data.dataloader import event_set
from model.CAMEF import CAMEF, test as evaluate, test_save


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RoBERTa-based CAMEF checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth or lastest_model.pth.")
    parser.add_argument("--seq-len", type=int, default=35, help="Input sequence length used during training.")
    parser.add_argument("--pred-len", type=int, default=35, help="Prediction horizon length used during training.")
    parser.add_argument("--event-id", type=int, default=0, help="Event id used by the dataloader.")
    parser.add_argument("--series-id", type=str, default="SP500", help="Series csv basename under data/series.")
    parser.add_argument("--event-dir", type=str, default="data/event", help="Event text data directory.")
    parser.add_argument("--series-dir", type=str, default="data/series", help="Time-series csv directory.")
    parser.add_argument("--bert-model", type=str, default="FacebookAI/roberta-base", help="RoBERTa model id or local path.")
    parser.add_argument("--gpt-model", type=str, default="openai-community/gpt2", help="GPT-2 model id or local path.")
    parser.add_argument(
        "--moment-model",
        type=str,
        default="AutonLab/MOMENT-1-large",
        help="MOMENT model id or local checkpoint path, such as AutonLab/MOMENT-1-large. It must match the checkpoint architecture.",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Evaluation batch size.")
    parser.add_argument("--window", type=int, default=500, help="Text encoder window size used during training.")
    parser.add_argument("--stride", type=int, default=400, help="Text encoder stride used during training.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before 6:2:2 splitting.")
    parser.add_argument("--split-by-time", action="store_true", help="Use chronological 6:2:2 train/test/validation split.")
    parser.add_argument(
        "--split",
        choices=("train", "test", "vali"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--save-samples-dir",
        type=str,
        default=None,
        help="Optional directory to save per-sample inputs, targets, and predictions.",
    )
    return parser.parse_args()


def main():
    warnings.simplefilter("ignore")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="torch.utils._pytree._register_pytree_node is deprecated",
    )
    torch.autograd.set_detect_anomaly(True)

    args = parse_args()

    dataset = event_set(
        args.seq_len,
        args.pred_len,
        event_id=args.event_id,
        series_id=args.series_id,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        scale=True,
        event_dir=args.event_dir,
        series_dir=args.series_dir,
        split_by_time=args.split_by_time,
    )

    model = CAMEF(
        bert=args.bert_model,
        moment=args.moment_model,
        gpt=args.gpt_model,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d=dataset.d,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    model.load_model_combined(args.checkpoint)

    loader = {
        "train": dataset.train_loader,
        "test": dataset.test_loader,
        "vali": dataset.vali_loader,
    }[args.split]

    if args.save_samples_dir:
        metrics = test_save(model, loader, dataset.scaler, output_folder=args.save_samples_dir)
    else:
        metrics = evaluate(model, loader)

    avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss = metrics
    print(
        "Evaluation results: "
        f"Combined={avg_combined_loss}, "
        f"MSE={avg_mse_loss}, "
        f"MAE={avg_mae_loss}, "
        f"Contrastive={avg_contrastive_loss}"
    )


if __name__ == "__main__":
    main()
