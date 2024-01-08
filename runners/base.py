# def predict_onestep(self):
#     # ! test onestep
#     self.res_true = []
#     self.res_pred = []

#     # print(len(self.test_loader.dataset))
#     running_tloss = 0.0
#     self.model.eval()
#     with torch.no_grad():
#         for i, (x_batch, y_batch) in enumerate(self.test_loader):
#             # * input
#             x_batch = x_batch.to(self.device)
#             y_batch = y_batch.unsqueeze(1).to(self.device)

#             # * forward
#             y_pred = self.model(x_batch)
#             loss = self.loss_fn(y_batch, y_pred)
#             running_tloss += loss
#             print(y_batch.shape, y_pred.shape)

#             # * result
#             y_batch = np.squeeze(y_batch.detach().numpy())
#             y_pred = np.squeeze(y_pred.detach().numpy())
#             # print(y_batch.shape, y_pred.shape)
#             self.res_true.extend(y_batch)
#             self.res_pred.extend(y_pred)
#             # print(len(self.res_true))

#     avg_tloss = running_tloss / len(self.test_loader)
#     print(f"test loss: {avg_tloss}")

#     self.res_df = pd.DataFrame(
#         {
#             "true": self.res_true,
#             "pred": self.res_pred,
#         }
#     )

# def predict_multistep(self):
#     test_data = torch.Tensor(self.data_dic["test"]).to(self.device)
#     # ! test multistep
#     self.res_true = []
#     self.res_pred = []
#     # print(len(self.test_loader.dataset))
#     running_tloss = 0.0
#     cnt = 0
#     self.model.eval()
#     with torch.no_grad():
#         for i in range(test_data.shape[0] - self.seq_len):
#             cnt += 1
#             # * input
#             x = test_data[i : i + self.seq_len, :]

#             # * forward
#             y_pred = self.model(x.view(1, self.seq_len, -1)).squeeze()
#             y_true = test_data[i + self.seq_len, -1]
#             loss = self.loss_fn(y_true, y_pred)
#             running_tloss += loss
#             # print(y_pred, y_true, loss)

#             # * result
#             y_pred = y_pred.detach().numpy().item()
#             y_true = y_true.detach().numpy().item()
#             self.res_true.append(y_true)
#             self.res_pred.append(y_pred)

#             # print("true: ", self.res_true)
#             # print("pred: ", self.res_pred)

#             # print(len(self.res_true))
#             # print(i + self.seq_len)
#             test_data[i + self.seq_len, -1] = torch.Tensor([y_pred])

#     avg_tloss = running_tloss / cnt
#     print(f"test loss: {avg_tloss}")

#     self.res_df = pd.DataFrame(
#         {
#             "true": self.res_true,
#             "pred": self.res_pred,
#         }
#     )


# def get_result(self, isLoss=True):
#     if isLoss:
#         return self.losses_df
#     else:
#         return self.res_df

# def inverse_res_df(self):
#     scaler = self.scaler_dic["scaler"]
#     columns = self.scaler_dic["columns"]
#     for col_name, col in self.res_df.items():
#         inversed_arr = scaler.inverse_transform(col.to_numpy())
#         self.res_df[col_name] = inversed_arr
#         # inversed_df = pd.DataFrame(inversed_arr, columns=columns)
#     display(self.res_df)

# def test(self):
#     print(f"\n>>>>>>>> func: {inspect.stack()[0][3]} <<<<<<<<")
#     for i, (x_batch, y_batch) in enumerate(self.train_loader):
#         if i >= 1:
#             break

#         print(f"x_batch: {x_batch.shape}, y_batch: {y_batch.shape}")
#         y_pred = self.model(x_batch)
#         print(f"y_pred: {y_pred.shape}")
#         loss = self.loss_fn(y_pred, y_batch)
#         print(f"loss: {loss}")
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# class SubRunner(Runner):
#     # super(SubRunner, self).__init__()
#     def run_building(self, df):
#         print("test")


# subrunner = SubRunner(model, optimizer, loss_fn)
