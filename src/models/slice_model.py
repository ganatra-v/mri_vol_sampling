import torch
import torch.nn as nn
from torchvision.models import resnet18
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

class SliceModel(nn.Module):
    def __init__(self, args):
        super(SliceModel, self).__init__()
        self.args = args
        self.model = resnet18()

        # Modify the first conv layer to accept n_channels input channels
        outchannels, kernelsize, stride, padding = self.model.conv1.out_channels, self.model.conv1.kernel_size, self.model.conv1.stride, self.model.conv1.padding
        self.model.conv1 = nn.Conv2d(args.n_channels, outchannels, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        num_ftrs = self.model.fc.in_features
        self.feature_dim = num_ftrs
        self.model.fc = nn.Identity()  # Remove final classification layer

        self.slice_classifier = nn.Linear(num_ftrs, 1)  # Binary classification

        if args.input_data_format == "slices+volumes":
            self.attention = nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Softmax(dim=1)
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
    
    def train_model(self, trainloader):
        self.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9]))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        vol_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in range(self.args.epochs):
            self.train()
            epoch_loss = 0.0
            for i, (inputs, slice_labels, vol_labels) in tqdm(enumerate(trainloader)):
                    # inputs has variable length inputs such as 5 x320 x 320, 32 x 320 x 320, 60 x 320 x 320, stack them into -1 x 1 x 320 x 320
                slices = torch.vstack([inp.unsqueeze(1) for inp in inputs])  # (total_slices, 1, height, width)
                slice_labels = torch.hstack(slice_labels).unsqueeze(1).float()  # (total_slices, 1)

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
                    for inp, vol_label in zip(inputs, vol_labels):
                        n_slices = inp.shape[0]
                        # reuse slice features from slice_forward
                        vol_slice_features = slice_features[start_idx:start_idx + n_slices]  # (n_slices, feature_dim)
                        start_idx += n_slices
                        # Compute attention weights
                        attn_weights = self.attention(vol_slice_features)  # (n_slices, 1)
                        # Weighted sum of slice features
                        vol_feature = torch.sum(attn_weights * vol_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_logit = self.vol_forward(vol_feature)  # (1, 1)
                        vol_preds.append(vol_logit)

                        vol_label = vol_label.to(device)
                        vol_labels.append(vol_label)
                    vol_preds = torch.vstack(vol_preds)  # (batch_size, 1)
                    vol_labels = torch.tensor(vol_labels).unsqueeze(1).float()  # (batch_size, 1)
                    vol_loss = vol_criterion(vol_preds, vol_labels)
                    total_loss += vol_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                self.eval_model(trainloader)  # evaluate on training set each iteration
            
            avg_loss = epoch_loss / len(trainloader)
            logging.info(f"epoch {epoch+1}/{self.args.epochs}, loss: {avg_loss:.4f}")
        logging.info("Finished Slice Model Training")
        torch.save(self.state_dict(), f"{self.args.outdir}/final_model.pth")
    
    def eval_metrics(self, labels, preds, threshold=0.5):
        roc = roc_auc_score(labels, preds)
        bin_preds = (preds >= threshold).astype(int)
        accuracy = accuracy_score(labels, bin_preds)
        precision = precision_score(labels, bin_preds, zero_division=0)
        recall = recall_score(labels, bin_preds, zero_division=0)
        f1 = f1_score(labels, bin_preds, zero_division=0)
        logging.info(f"acc: {accuracy:.4f}, prec: {precision:.4f}, rec: {recall:.4f}, f1: {f1:.4f}, roc_auc: {roc:.4f}")
        return accuracy, precision, recall, f1, roc
    
    def eval_model(self, dataloader):
        self.eval()
        all_slice_labels = []
        all_slice_preds = []

        if self.args.input_data_format == "slices+volumes":
            all_vol_labels = []
            all_vol_preds = []
        
        with torch.no_grad():
            for i, (inputs, slice_labels, vol_labels) in enumerate(dataloader):
                slices = torch.vstack([inp.unsqueeze(1) for inp in inputs])  # (total_slices, 1, height, width)
                slice_labels = torch.hstack(slice_labels).unsqueeze(1).float()  # (total_slices, 1)
                slice_logits, slice_features = self.slice_forward(slices)
                slice_probs = torch.sigmoid(slice_logits)

                all_slice_labels.append(slice_labels.cpu())
                all_slice_preds.append(slice_probs.cpu())

                if self.args.input_data_format == "slices+volumes":
                    vol_preds = []
                    start_idx = 0
                    for inp, vol_label in zip(inputs, vol_labels):
                        n_slices = inp.shape[0]
                        vol_slice_features = slice_features[start_idx:start_idx + n_slices]  # (n_slices, feature_dim)
                        start_idx += n_slices
                        attn_weights = self.attention(vol_slice_features)  # (n_slices, 1)
                        vol_feature = torch.sum(attn_weights * vol_slice_features, dim=0, keepdim=True)  # (1, feature_dim)
                        vol_logit = self.vol_forward(vol_feature)  # (1, 1)
                        vol_preds.append(vol_logit)
                    vol_preds = torch.vstack(vol_preds)  # (batch_size, 1)
                    vol_probs = torch.sigmoid(vol_preds)

                    all_vol_labels.append(torch.tensor(vol_labels).unsqueeze(1).float().cpu())
                    all_vol_preds.append(vol_probs.cpu())
        all_slice_labels = torch.vstack(all_slice_labels).numpy()
        all_slice_preds = torch.vstack(all_slice_preds).numpy()
        logging.info("slice-level metrics.....................")
        self.eval_metrics(all_slice_labels, all_slice_preds)
        if self.args.input_data_format == "slices+volumes":
            all_vol_labels = torch.vstack(all_vol_labels).numpy()
            all_vol_preds = torch.vstack(all_vol_preds).numpy()
            logging.info("volume-level metrics.....................")
            self.eval_metrics(all_vol_labels, all_vol_preds)