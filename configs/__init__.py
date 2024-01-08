cfg_lstm = dict(bidirectional=False)


# cfg_tsmixer = dict(
#     filename="electricity",
#     seq_len=48,
#     pred_len=24,
#     batch_size=64,
#     feature_type="MS",
#     target="3",
#     model="tsmixer",
#     norm_type="B",
#     activation="relu",
#     dropout=0.05,
#     n_block=2,
#     hidden_dims=[48 * 4],
#     output_dims=[128],
#     optim_name="Adam",
#     lossfn_name="MSE",
#     lr=0.001,
#     epochs=10,
#     options=dict(
#         output_dims=[128],
#     ),
# )


# cfg_lstm = dict(
#     filename="electricity",
#     seq_len=48,
#     pred_len=24,
#     batch_size=64,
#     feature_type="MS",
#     target="3",
#     model="tslstm",
#     norm_type="B",
#     activation="relu",
#     dropout=0.05,
#     n_block=2,
#     hidden_dims=[48 * 4],
#     optim_name="Adam",
#     lossfn_name="MSE",
#     lr=0.001,
#     epochs=10,
#     options=dict(bidirectional=False),
# )


# cfg_tide = dict(
#     filename="electricity",
#     seq_len=48,
#     pred_len=24,
#     batch_size=16,
#     feature_type="M",
#     target=None,
#     model="tide",
#     norm_type="B",
#     activation="relu",
#     dropout=0.05,
#     n_block=2,
#     hidden_dims=[128],
#     options=dict(
#         use_layer_norm=True,
#         output_dims=[128],
#     ),
#     optim_name="Adam",
#     lossfn_name="MSE",
#     lr=0.001,
#     epochs=10,
# )
