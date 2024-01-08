import argparse

from easydict import EasyDict as edict
from pprint import pprint

from data_provider import TSFDataLoader
import torch.optim as optim
import models, losses, runners

# from utils import *
import configs


def parse_args():
    parser = argparse.ArgumentParser(description="TSMixer for Time Series Forecasting")
    # base
    parser.add_argument(
        "--data_dir", type=str, default="./data/base", help="set data directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # data
    parser.add_argument("--filename", default="electricity", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_len", default=48, type=int)
    parser.add_argument("--pred_len", default=24, type=int)
    parser.add_argument(
        "--feature_type",
        type=str,
        default="MS",
        choices=["S", "M", "MS"],
        help=(
            "forecasting task, options:[M, S, MS]; M:multivariate predict"
            " multivariate, S:univariate predict univariate, MS:multivariate"
            " predict univariate"
        ),
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="2",
        help="target feature in S or MS task",
    )
    # model
    parser.add_argument(
        "--model", type=str, default="lstm", help="put the name of model"
    )
    parser.add_argument("--hidden_dims", type=list, default=[24])
    parser.add_argument("--output_dims", type=list, default=[1])
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    # run
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--loss_fn", type=str, default="MSE")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--runner", type=str, default="tsfrunner")

    # save
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints")
    parser.add_argument("--model_path", type=str, default="./save")
    parser.add_argument("--output_path", type=str, default="./output")
    # parser.add_argument('--', type=, default=)

    args = parser.parse_args()

    return args


def main():
    # * args
    args = parse_args()
    pprint(args)

    # * data load
    tsfdataloader = TSFDataLoader(
        args.filename,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target_column,
        limit=[None, 1500],
        print_option=False,
    )
    dataloaders = tsfdataloader.get_dataloaders()
    df = tsfdataloader.get_df()
    print(df.shape)

    # * data test
    (x_batch, y_batch), n_feature, target_idx = tsfdataloader.get_data_info()
    args.n_feature = n_feature
    args.target_idx = target_idx

    cfg = edict(getattr(configs, f"cfg_{args.model}"))
    pprint(cfg)

    # * define
    build_model = getattr(models, args.model).build_model
    model = build_model(
        args.seq_len,
        args.n_feature,
        args.hidden_dims,
        args.dropout,
        args.n_block,
        args.pred_len,
        args.target_idx,
        cfg,
    )
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    loss_fn = losses.build_loss_fn(args.loss_fn)

    run_equipment = dict(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataloaders=dataloaders,
        seed=args.seed,
        seq_len=args.seq_len,
        ckpt_path=args.ckpt_path,
        model_path=args.model_path,
        # output_path=args.output_path,
    )

    # runner = runners.TSFRunner(run_equipment, epochs=args.epochs)
    runner = getattr(runners, args.runner).build_runner(
        run_equipment, epochs=args.epochs
    )

    # * run
    res_train_df, res_test_df = runner.run()

    # * eval

    # * save


if __name__ == "__main__":
    main()
