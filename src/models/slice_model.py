import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd

class SliceModel(nn.Module):
    def __init__(self, args):
        super(SliceModel, self).__init__()
        self.args = args
        if args.arch == "resnet18":
            self.model = resnet18()
        elif args.arch == "resnet34":
            self.model = resnet34()
        elif args.arch == "resnet50":
            self.model = resnet50()

        # Modify the first conv layer to accept n_channels input channels
        outchannels, kernelsize, stride, padding = self.model.conv1.out_channels, self.model.conv1.kernel_size, self.model.conv1.stride, self.model.conv1.padding
        self.model.conv1 = nn.Conv2d(args.n_channels, outchannels, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        num_ftrs = self.model.fc.in_features
        self.feature_dim = num_ftrs
        self.model.fc = nn.Identity()  # Remove final classification layer

        self.slice_classifier = nn.Linear(num_ftrs, 1)  # Binary classification

        if args.input_data_format == "slices+volumes":
            self.attention = nn.Sequential(
                self.slice_classifier,
                nn.Softmax(dim=0)
            )
            self.vol_classifier = nn.Linear(num_ftrs, 1)  # Binary classification

    def slice_forward(self, x):
        # x is (n_batches, n_channels, height, width)
        # the model expects (n_batches * n_slices, n_channels, height, width)
        features = self.model(x)  # (n_batches * n_slices, feature_dim)
        slice_logits = self.slice_classifier(features)  # (n_batches * n_slices, 1)

        return slice_logits, features
        
    def vol_forward(self, x):
        x = self.vol_classifier(x)  # (n_batches, 1)
        return x
    
    def train_model(self, trainloader, valloader):
        self.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7.5]).cuda())
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        vol_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.75]).cuda())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.args.epochs // 4 * 3], gamma=0.2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_acc , best_epoch = 0, -1

        for epoch in range(self.args.epochs):
            self.train()
            epoch_loss = 0.0
            for i, (_, inputs, slice_labels, vol_labels_) in tqdm(enumerate(trainloader)):
                    # inputs has variable length inputs such as 5 x320 x 320, 32 x 320 x 320, 60 x 320 x 320, stack them into -1 x 1 x 320 x 320
                slices = torch.vstack([inp.unsqueeze(1) for inp in inputs])  # (total_slices, 1, height, width)

                # use torch.cat instead of torch.stack to avoid extra dimension
                slice_labels = slice_labels.view(-1, 1).float()  # (total_slices, 1)

                slices, slice_labels = slices.to(device), slice_labels.to(device)

                assert slices.shape[0] == slice_labels.shape[0], "Number of slices and labels must match"
                optimizer.zero_grad()
                slice_logits, slice_features = self.slice_forward(slices)
                slice_loss = criterion(slice_logits, slice_labels)
                total_loss = self.args.slice_loss_lam * slice_loss

                if self.args.input_data_format == "slices+volumes":
                    # Volume-level training
                    vol_preds = []
                    vol_labels = []
                    start_idx = 0
                    for inp, vol_label in zip(inputs, vol_labels_):
                        n_slices = inp.shape[0]
                        # reuse slice features from slice_forward
                        vol_slice_features = slice_features[start_idx:start_idx + n_slices]  # (n_slices, feature_dim)
                        start_idx += n_slices
                        # Compute attention weights
                        attn_weights = self.attention(vol_slice_features)  # (n_slices, 1)
                        # Weighted sum of slice features
                        topk_scores, topk_indices = torch.topk(attn_weights.squeeze(), k=5, largest=True)
                        topk_slice_features = vol_slice_features[topk_indices]
                        topk_attn_weights = topk_scores.unsqueeze(1)
                        vol_feature = torch.sum(topk_attn_weights * topk_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        # normalize vol_feature
                        vol_feature = vol_feature / torch.sum(topk_attn_weights)
                        # vol_feature = torch.sum(attn_weights * vol_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_logit = self.vol_forward(vol_feature)  # (1, 1)
                        vol_preds.append(vol_logit)

                        vol_label = vol_label.to(device)
                        vol_labels.append(vol_label)
                    vol_preds = torch.vstack(vol_preds).cuda()  # (batch_size, 1)
                    vol_labels = torch.tensor(vol_labels).unsqueeze(1).float().cuda()  # (batch_size, 1)
                    vol_loss = vol_criterion(vol_preds, vol_labels)
                    total_loss += vol_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                print(f"step {i+1}/{len(trainloader)}, loss: {epoch_loss/(i+1):.4f}")
            acc, prec, rec, f1, roc, sens, spec = self.eval_model(valloader)  # evaluate on validation set each iteration
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                torch.save(self.state_dict(), f"{self.args.outdir}/best_model.pth")
                logging.info(f"New best model saved at epoch {best_epoch} with accuracy {best_acc:.4f}")
            
            avg_loss = epoch_loss / len(trainloader)
            logging.info(f"epoch {epoch+1}/{self.args.epochs}, loss: {avg_loss:.4f}, lr: {lr:.7f}")
        logging.info("Finished Slice Model Training")
        torch.save(self.state_dict(), f"{self.args.outdir}/final_model.pth")
    
    def eval_metrics(self, labels, preds, threshold=0.5):
        roc = roc_auc_score(labels, preds)
        bin_preds = (preds >= threshold).astype(int)
        accuracy = accuracy_score(labels, bin_preds)
        precision = precision_score(labels, bin_preds, zero_division=0)
        recall = recall_score(labels, bin_preds, zero_division=0)
        f1 = f1_score(labels, bin_preds, zero_division=0)
        sens = recall
        spec = recall_score(labels, bin_preds, pos_label=0)
        logging.info(f"acc: {accuracy:.4f}, prec: {precision:.4f}, rec: {recall:.4f}, f1: {f1:.4f}, roc_auc: {roc:.4f}, sens: {sens:.4f}, spec: {spec:.4f}")
        return accuracy, precision, recall, f1, roc, sens, spec
    
    def eval_model(self, dataloader, save_topk_slices=False, save_preds=False):
        self.eval()
        filenames = []
        all_slice_labels = []
        all_slice_preds = []

        if self.args.input_data_format == "slices+volumes":
            volume_names = []
            all_vol_labels = []
            all_vol_preds = []
            if save_topk_slices:
                topk_slices = []        
        
        with torch.no_grad():
            for i, (filename, inputs, slice_labels, vol_labels) in enumerate(dataloader):
                inputs, slice_labels, vol_labels = inputs.cuda(), slice_labels.cuda(), vol_labels.cuda()
                slices = torch.vstack([inp.unsqueeze(1) for inp in inputs])  # (total_slices, 1, height, width)
                slice_labels = slice_labels.unsqueeze(1).float()  # (total_slices, 1)
                slice_logits, slice_features = self.slice_forward(slices)
                slice_probs = torch.sigmoid(slice_logits)

                all_slice_labels.append(slice_labels.cpu())
                all_slice_preds.append(slice_probs.cpu())

                # repeat each filename for number of slices without using a loop
                for f in filename:
                    filenames.extend([f] * slice_labels.shape[-1])

                if self.args.input_data_format == "slices+volumes":
                    volume_names.extend(filename)
                    vol_preds = []
                    start_idx = 0
                    for inp, vol_label in zip(inputs, vol_labels):
                        n_slices = inp.shape[0]
                        vol_slice_features = slice_features[start_idx:start_idx + n_slices]  # (n_slices, feature_dim)
                        start_idx += n_slices
                        attn_weights = self.attention(vol_slice_features)  # (n_slices, 1)

                        topk_scores, topk_indices = torch.topk(attn_weights.squeeze(), k=5, largest=True)
                        topk_slice_features = vol_slice_features[topk_indices]
                        topk_attn_weights = topk_scores.unsqueeze(1)
                        vol_feature = torch.sum(topk_attn_weights * topk_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_feature = vol_feature / torch.sum(topk_attn_weights)
                        # vol_feature = torch.sum(attn_weights * vol_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_logit = self.vol_forward(vol_feature)  # (1, 1)
                        vol_preds.append(vol_logit)

                        if save_topk_slices:
                            topk_slices.append(topk_indices.cpu().numpy())
                    vol_preds = torch.vstack(vol_preds)  # (batch_size, 1)
                    vol_probs = torch.sigmoid(vol_preds)

                    all_vol_labels.append(torch.tensor(vol_labels).unsqueeze(1).float().cpu())
                    all_vol_preds.append(vol_probs.cpu())
        all_slice_labels = torch.vstack(all_slice_labels).numpy().reshape(-1)
        all_slice_preds = torch.vstack(all_slice_preds).numpy().reshape(-1)
        if save_preds:
            slice_data = {
                "filenames": filenames,
                "slice_labels": all_slice_labels,
                "slice_preds": all_slice_preds
            }
            slice_data = pd.DataFrame(slice_data)
            slice_data.to_csv(f"{self.args.outdir}/slice_predictions.csv", index=False)
        logging.info("slice-level metrics.....................")
        acc, prec, rec, f1, roc, sens, spec = self.eval_metrics(all_slice_labels, all_slice_preds)
        
        if self.args.input_data_format == "slices+volumes":
            all_vol_labels = torch.vstack(all_vol_labels).numpy().reshape(-1)
            all_vol_preds = torch.vstack(all_vol_preds).numpy().reshape(-1)
            logging.info("volume-level metrics.....................")
            acc, prec, rec, f1, roc, sens, spec = self.eval_metrics(all_vol_labels, all_vol_preds)
            if save_topk_slices:
                topk_slices = np.array(topk_slices)
                topk_data = {
                    "vol_labels": all_vol_labels,
                    "vol_preds": all_vol_preds,
                    "topk_slices": topk_slices
                }
                np.save(f"{self.args.outdir}/topk_slices.npy", topk_data)
            if save_preds:
                vol_data = {
                    "filenames": volume_names,
                    "vol_labels": all_vol_labels,
                    "vol_preds": all_vol_preds
                }
                vol_data = pd.DataFrame(vol_data)
                vol_data.to_csv(f"{self.args.outdir}/volume_predictions.csv", index=False)
        return acc, prec, rec, f1, roc, sens, spec