import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertModel, BertForMaskedLM, GPT2Model, LongformerTokenizer, LongformerModel, \
    RobertaTokenizer, RobertaModel
from momentfm import MOMENTPipeline
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import warnings
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging

# 屏蔽所有警告
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.utils._pytree._register_pytree_node is deprecated")


class GPT4MTS(nn.Module):
    def __init__(self, bert, moment, gpt, seq_len, pred_len, d, window, stride, batch_size):
        super(GPT4MTS, self).__init__()

        self.seq_len, self.pred_len = seq_len, pred_len
        self.window, self.stride = window, stride
        self.d = d
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gpt_model = GPT2Model.from_pretrained(gpt)
        self.gpt_model.drop.to(self.device)
        for layer in self.gpt_model.h:
            layer.to(self.device)
        self.gpt_model.ln_f.to(self.device)


        self.bert_model = RobertaModel.from_pretrained(bert).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(bert)



        self.bert_linear =  nn.Sequential(
            nn.Linear(768, 1024, bias=True),
            nn.GELU(),
            nn.Linear(1024, 768, bias=True),
            nn.ReLU()
        ).to(self.device)


        self.moment_model = MOMENTPipeline.from_pretrained(
            pretrained_model_name_or_path=moment,
            model_kwargs={
                'task_name': 'embedding',
            },
            cache_dir=r'moment_model',
            local_files_only=True,
        ).to(self.device)
        self.moment_model.init()
        # print(self.moment_model)

        # serirs -> 主模型
        self.embed_project = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.GELU(),
            nn.Linear(1024, 1024, bias=True),
            nn.GELU(),
            nn.Linear(1024, 768, bias=True)  # 5 第三层
        ).to(self.device)

        self.fuse_project = nn.Sequential(
            nn.Linear(2 * 768, 1024),  # First layer
            nn.GELU(),  # GELU activation function
            nn.Linear(1024, 768),  # Second layer
        ).to(device=self.device)



        self.output_project = nn.Sequential(
            nn.Linear(self.gpt_model.ln_f.normalized_shape[0], self.gpt_model.ln_f.normalized_shape[0], bias=True),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(self.gpt_model.ln_f.normalized_shape[0], self.gpt_model.ln_f.normalized_shape[0], bias=True),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(self.gpt_model.ln_f.normalized_shape[0], self.gpt_model.ln_f.normalized_shape[0], bias=True),
            nn.GELU(),
            # nn.Dropout(p=0.1),
            nn.Linear(self.gpt_model.ln_f.normalized_shape[0], self.gpt_model.ln_f.normalized_shape[0], bias=True),
            nn.GELU(),
            nn.Dropout(p=0.1),

            nn.Linear(self.gpt_model.ln_f.normalized_shape[0], self.d * pred_len, bias=True)
        ).to(self.device)

        self.residual_project = nn.Sequential(
            nn.Linear(self.seq_len * self.d, self.seq_len * self.d, bias=True),
            nn.GELU(),
            nn.Linear(self.seq_len * self.d, self.seq_len * self.d, bias=True),
            nn.GELU(),
            nn.Linear(self.seq_len * self.d, 768, bias=True)
        ).to(self.device)



    # 单条数据的前向传播
    def forward(self, text, seq_data):
        seq_data = seq_data.float().to(self.device)
        text_embedding = self.text_embedding(text).to(self.device)  # torch.Size([768])
        series_embedding = self.moment_model(seq_data.unsqueeze(0)).embeddings  # torch.Size([batch_size, 1024])
        series_embedding = self.embed_project(series_embedding[0])  # torch.Size([768])
        series_embedding = series_embedding + self.residual_project(seq_data.reshape(-1))
        input_embedding = torch.cat(
            (text_embedding.unsqueeze(0).unsqueeze(0), series_embedding.unsqueeze(0).unsqueeze(0)), dim=1)

        output = self.gpt_model.drop(input_embedding).to(self.device)
        for layer in self.gpt_model.h:
            layer.to(self.device)
            if isinstance(output, tuple):
                output = output[0]
            output = layer(output)
        self.gpt_model.ln_f.to(self.device)
        output = self.gpt_model.ln_f(output[0])



        result = self.output_project(output.reshape(1, 2 * self.gpt_model.ln_f.normalized_shape[0])).reshape(1, self.d,
                                                                                                             self.pred_len)
        return result

    # 单条数据获取text embedding, batch的不用写，因为输入的token不等长，必须截断或padding
    def text_embedding(self, text):

        encoded_input = self.tokenizer(text, return_tensors='pt')['input_ids']
        embeddings = []
        start = 0
        while start < encoded_input.shape[-1]:
            tokens = encoded_input[0][start:min(start + self.window, encoded_input.shape[-1])]
            start += self.stride


            ## full roberta
            embeddings.append(
                torch.mean(self.bert_model(tokens.unsqueeze(0).to(self.device)).last_hidden_state.squeeze(0), dim=0).unsqueeze(0).cpu())

            torch.cuda.empty_cache()
        return torch.mean(torch.cat(embeddings, dim=0), dim=0).float().to(self.device)


    # batch数据
    def predict_batch(self, batch_text, batch_seq):

        batch_text_embeddings = []
        for text in batch_text:
            batch_text_embeddings.append(self.text_embedding(text).unsqueeze(0))
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)  # torch.Size([batch_size, 768])

        batch_seq = batch_seq.float().to(self.device)

        batch_series_embeddings = self.moment_model(batch_seq).embeddings  # torch.Size([batch_size, 1024])
        batch_series_embeddings = self.embed_project(batch_series_embeddings)  # torch.Size([batch_size, 768])
        batch_series_embeddings = batch_series_embeddings + self.residual_project(
            batch_seq.reshape(-1, self.seq_len * self.d))

        # batch_series_embeddings = self.residual_project(batch_seq.reshape(self.batch_size,-1))

        batch_input_embeddings = torch.stack((batch_text_embeddings, batch_series_embeddings),
                                             dim=1)  # batch_size, 2 ,768

        batch_output = self.gpt_model.drop(batch_input_embeddings).to(self.device)
        for layer in self.gpt_model.h:
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            batch_output = layer(batch_output)
        batch_output = self.gpt_model.ln_f(batch_output[0])
        output_data = []
        for output in batch_output:
            output_data.append(
                self.output_project(output.reshape(1, 2 * self.gpt_model.ln_f.normalized_shape[0])).reshape(1, self.d,
                                                                                                            self.pred_len))
        output = torch.cat(output_data, dim=0)
        # batch_output = self.bert_pooler(self.bert_main(batch_input_embeddings).last_hidden_state)
        # batch_output = self.output_project(batch_output).reshape(-1 , self.d, self.pred_len)
        return output, batch_text_embeddings

    def predict_batch_contrastive(self, batch_text, batch_sent_reports, batch_negative_type_reports, batch_seq):

        # Unfreeze only the parameters in the output layers of each encoder layer

        # 1. Embedding extraction for text
        batch_text_embeddings = []
        for text in batch_text:
            batch_text_embeddings.append(self.bert_linear(self.text_embedding(text)).unsqueeze(0))
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)  # [batch_size, 768]

        # 2. Embeddings of the Sent Reports
        batch_sent_report_embeddings = []
        for sent_reports in batch_sent_reports:
            report_embeddings = []
            for report in sent_reports:
                report_embeddings.append(self.bert_linear(self.text_embedding(report)).unsqueeze(0))
            batch_sent_report_embeddings.append(report_embeddings)
        batch_sent_report_embeddings = [torch.cat(e, dim=0) for e in batch_sent_report_embeddings]
        batch_sent_report_embeddings = torch.stack(batch_sent_report_embeddings, dim=0)  # [batch_size, 10, 768]

        batch_sent_report_embeddings = torch.cat([batch_text_embeddings.unsqueeze(1), batch_sent_report_embeddings],
                                                 dim=1)
        # 3. Embeddings of the Negative Reports
        batch_negative_type_report_embeddings = []
        for negative_reports in batch_negative_type_reports:
            negative_report_embeddings = []
            for negative_report in negative_reports:
                negative_report_embeddings.append(
                    self.bert_linear(self.text_embedding(negative_report)).unsqueeze(0))
            batch_negative_type_report_embeddings.append(negative_report_embeddings)
        batch_negative_type_report_embeddings = [torch.cat(e, dim=0) for e in batch_negative_type_report_embeddings]
        batch_negative_type_report_embeddings = torch.stack(batch_negative_type_report_embeddings,
                                                            dim=0)  # [batch_size, 5, 768]

        # 4. Time-series embeddings
        batch_seq = batch_seq.float().to(self.device)
        batch_series_embeddings = self.moment_model(x_enc=batch_seq).embeddings  # [batch_size, 1024]
        batch_series_embeddings = self.embed_project(batch_series_embeddings)  # [batch_size, 768]
        batch_series_embeddings = batch_series_embeddings + self.residual_project(
            batch_seq.reshape(-1, self.seq_len * self.d))
        # # 5. Multi-head attention to model interaction between text and series embeddings
        batch_input_embeddings = torch.stack((batch_text_embeddings, batch_series_embeddings),
                                             dim=1)  # [batch_size, 2, 768]
        # batch_input_embeddings = torch.mean(batch_input_embeddings, dim=1)  # [batch_size, 768]

        ## Full Model
        batch_input_embeddings = self.fuse_project(
            batch_input_embeddings.reshape(batch_input_embeddings.shape[0], -1))
        ## Ablation (No Textual Modality)
        # batch_input_embeddings = batch_series_embeddings.reshape(batch_input_embeddings.shape[0], -1)

        # Process batch through GPT model
        batch_output = self.gpt_model.drop(batch_input_embeddings).unsqueeze(1).to(self.device)
        for layer in self.gpt_model.h:
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            batch_output = layer(batch_output)
        batch_output = self.gpt_model.ln_f(batch_output[0])  # Final output after processing by GPT

        # Concatenate the tensors along the last dimension
        # combined_tensor = torch.cat((batch_output, batch_text_embeddings.unsqueeze(1)), dim=1)
        # batch_output = torch.mean(combined_tensor, dim=1).unsqueeze(1)

        output_data = []
        for output in batch_output:
            output_data.append(
                self.output_project(output.reshape(1, self.gpt_model.ln_f.normalized_shape[0])).reshape(1, self.d,
                                                                                                        self.pred_len))
        output = torch.cat(output_data, dim=0)


        return output, batch_output, batch_series_embeddings, batch_text_embeddings, batch_sent_report_embeddings, batch_negative_type_report_embeddings

    def predict_single_case(self, batch_text, batch_seq):
        # Unfreeze only the parameters in the output layers of each encoder layer
        # 1. Embedding extraction for text
        batch_text_embeddings = []
        for text in batch_text:
            batch_text_embeddings.append(self.bert_linear(self.text_embedding(text)).unsqueeze(0))
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)  # [batch_size, 768]

        # 2. Time-series embeddings
        batch_seq = batch_seq.float().to(self.device)
        batch_series_embeddings = self.moment_model(batch_seq).embeddings  # [batch_size, 1024]
        batch_series_embeddings = self.embed_project(batch_series_embeddings)  # [batch_size, 768]
        batch_series_embeddings = batch_series_embeddings + self.residual_project(
            batch_seq.reshape(-1, self.seq_len * self.d))
        # # 5. Multi-head attention to model interaction between text and series embeddings
        batch_input_embeddings = torch.stack((batch_text_embeddings, batch_series_embeddings),
                                             dim=1)  # [batch_size, 2, 768]
        # batch_input_embeddings = torch.mean(batch_input_embeddings, dim=1)  # [batch_size, 768]

        ## Full Model
        batch_input_embeddings = self.fuse_project(
            batch_input_embeddings.reshape(batch_input_embeddings.shape[0], -1))

        # Process batch through GPT model
        batch_output = self.gpt_model.drop(batch_input_embeddings).unsqueeze(1).to(self.device)
        for layer in self.gpt_model.h:
            if isinstance(batch_output, tuple):
                batch_output = batch_output[0]
            batch_output = layer(batch_output)
        batch_output = self.gpt_model.ln_f(batch_output[0])  # Final output after processing by GPT


        output_data = []
        for output in batch_output:
            output_data.append(
                self.output_project(output.reshape(1, self.gpt_model.ln_f.normalized_shape[0])).reshape(1, self.d,
                                                                                                        self.pred_len))
        output = torch.cat(output_data, dim=0)

        return output, batch_output, batch_series_embeddings, batch_text_embeddings

    def predict_one(self, text, seq_data):
        return self.forward(text, seq_data)

    def save_model_combined(self, save_path="model_checkpoint.pth", **kwargs):
        """
        Save a combined model with multiple components (e.g., GPT-2, MOMENT, RoBERTa, etc.), tokenizer,
        optimizer, and additional metadata.

        Parameters:
            self (torch.nn.Module): The PyTorch model containing GPT-2, MOMENT, RoBERTa, and custom layers.
            save_path (str): Path to save the checkpoint.
            **kwargs: Additional metadata to save, such as hyperparameters or configurations.
        """
        checkpoint = {
            'gpt_state_dict': self.gpt_model.state_dict(),
            'moment_state_dict': self.moment_model.state_dict(),
            'bert_state_dict': self.bert_model.state_dict(),
            'bert_linear_state_dict': self.bert_linear.state_dict(),
            'embed_project_state_dict': self.embed_project.state_dict(),
            'fuse_project_state_dict': self.fuse_project.state_dict(),
            'output_project_state_dict': self.output_project.state_dict(),
            'residual_project_state_dict': self.residual_project.state_dict(),
            'gpt_ln_f_state_dict': self.gpt_model.ln_f.state_dict(),
            'metadata': kwargs  # Save any additional metadata like hyperparameters
        }

        # Save the model checkpoint (ensure it's a file, not a folder)
        torch.save(checkpoint, save_path)
        print(f"Model saved successfully to {save_path}")

        # Save the tokenizer in a directory
        tokenizer_save_dir = os.path.join(os.path.dirname(save_path), 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_save_dir)
        print(f"Tokenizer saved successfully to {tokenizer_save_dir}")

    def load_model_combined(self, save_path="model_checkpoint.pth", optimizer=None, **kwargs):
        """
        Load a combined model (e.g., GPT-2, MOMENT, RoBERTa, etc.), tokenizer, and optimizer from a checkpoint.

        Parameters:
            self (torch.nn.Module): The empty model to load into.
            save_path (str): Path to the checkpoint.
        """
        checkpoint = torch.load(save_path)

        # Load individual models' state dicts
        self.gpt_model.load_state_dict(checkpoint['gpt_state_dict'])
        self.gpt_model.ln_f.load_state_dict(checkpoint['gpt_ln_f_state_dict'])
        self.moment_model.load_state_dict(checkpoint['moment_state_dict'])
        self.bert_model.load_state_dict(checkpoint['bert_state_dict'])
        self.bert_linear.load_state_dict(checkpoint['bert_linear_state_dict'])
        self.embed_project.load_state_dict(checkpoint['embed_project_state_dict'])
        self.fuse_project.load_state_dict(checkpoint['fuse_project_state_dict'])
        self.output_project.load_state_dict(checkpoint['output_project_state_dict'])
        self.residual_project.load_state_dict(checkpoint['residual_project_state_dict'])

        # Load the tokenizer
        save_folder_path = Path(save_path).parent
        tokenizer_path = os.path.join(save_folder_path,"tokenizer")
        self.tokenizer =RobertaTokenizer.from_pretrained(tokenizer_path)

        print(f"Model loaded successfully from {save_path}")
        return self
    # def predict_batch_mask(self, batch_seq, batch_pred):
    #     batch_seq_embedding = self.moment_model(batch_seq.float().to(self.device)).embeddings
    #     batch_pred_embedding = self.moment_model(batch_pred.float().to(self.device)).embeddings
    #
    #     batch_seq_embedding = self.embed_project(batch_seq_embedding).unsqueeze(1)
    #     batch_pred_embedding = self.embed_project(batch_pred_embedding).unsqueeze(1)
    #
    #     input_embedding = torch.cat((self.mask_embedding.repeat(min(self.batch_szie,batch_seq_embedding.shape[0]) ,1,1),batch_seq_embedding, batch_pred_embedding),dim=1)
    #
    #     output = self.bert_main(input_embedding).last_hidden_state
    #     output = self.bert_cls(output)
    #     return torch.mean(output, dim=1)


# def contrastive_loss_objective(batch_output, batch_sent_report_embeddings, batch_negative_type_report_embeddings,
#                      temperature=0.5):
#     # Normalize embeddings
#     batch_output_norm = F.normalize(batch_output, dim=-1)  # [7, 2, 768]
#     batch_sent_report_norm = F.normalize(batch_sent_report_embeddings, dim=-1)  # [7, 10, 768]
#     batch_negative_type_report_norm = F.normalize(batch_negative_type_report_embeddings, dim=-1)  # [7, 5, 768]
#
#     # Concatenate negative samples
#     all_negative_samples = torch.cat([batch_sent_report_norm, batch_negative_type_report_norm],
#                                      dim=1)  # [7, 15, 768]
#
#     # **Step 1**: Simplify batch_output_norm by combining the two modalities (time series and text).
#     # For example, take the mean of the two modalities to create a unified embedding for comparison.
#     batch_output_combined = batch_output_norm.mean(dim=1)  # [7, 768]
#
#     # **Step 2**: Compute similarity (dot product)
#     sim_positive = torch.matmul(batch_output_combined.unsqueeze(1),
#                                 all_negative_samples.transpose(1, 2)) / temperature  # [7, 1, 15]
#
#     sim_positive = sim_positive.squeeze(1)  # [7, 15]
#
#     # **Step 3**: Labels: first item in each row is the positive sample
#     labels = torch.zeros(batch_output_combined.size(0), dtype=torch.long).to(batch_output_combined.device)  # [7]
#
#     # **Step 4**: Contrastive loss using cross-entropy
#     loss = F.cross_entropy(sim_positive, labels)
#     return loss


def contrastive_loss_objective(batch_output, batch_sent_report_embeddings, batch_negative_type_report_embeddings,
                               margin=1.0, temperature=0.5):
    # Normalize embeddings
    batch_output_norm = F.normalize(batch_output.reshape(-1,1,768), dim=-1)  # [7, 2, 768]
    batch_sent_report_norm = F.normalize(batch_sent_report_embeddings, dim=-1)  # [7, 10, 768]
    batch_negative_type_report_norm = F.normalize(batch_negative_type_report_embeddings, dim=-1)  # [7, 5, 768]

    # Concatenate negative samples
    all_negative_samples = torch.cat([batch_sent_report_norm, batch_negative_type_report_norm], dim=1)  # [7, 15, 768]
    # print(all_negative_samples.shape)

    # **Step 1**: Simplify batch_output_norm by combining the two modalities (time series and text)
    # For example, take the mean of the two modalities to create a unified embedding for comparison.
    batch_output_combined = batch_output_norm.mean(dim=1)  # [7, 768]
    # batch_output_combined = batch_output_norm.unsqueeze(1) # [7, 768]

    # **Step 2**: Compute similarity (dot product) for cross-entropy contrastive loss
    sim_positive = torch.matmul(batch_output_combined.unsqueeze(1),
                                all_negative_samples.transpose(1, 2)) / temperature  # [7, 1, 15]
    sim_positive = sim_positive.squeeze(1)  # [7, 15]

    # **Step 3**: Labels: first item in each row is the positive sample
    labels = torch.zeros(batch_output_combined.size(0), dtype=torch.long).to(batch_output_combined.device)  # [7]

    # **Step 4**: Contrastive loss using cross-entropy
    ce_loss = F.cross_entropy(sim_positive, labels)

    # --- Additional Losses ---

    # **Triplet Loss**: Use the first report as the positive sample and the negative reports for negative samples
    positive_sample = batch_sent_report_norm[:, 0, :]  # Positive example for each batch
    negative_sample = all_negative_samples.mean(dim=1)  # Average of all negative samples

    # Triplet loss: anchor (batch_output_combined), positive (positive_sample), negative (negative_sample)
    pos_dist = F.pairwise_distance(batch_output_combined, positive_sample)
    neg_dist = F.pairwise_distance(batch_output_combined, negative_sample)

    triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()

    # **Hinge-based Contrastive Loss (Hadsell et al.)**
    # Encourage the positive pair to be close and the negative pair to be far apart by a margin
    # positive_loss = pos_dist.pow(2).mean()  # Positive pair distance
    # negative_loss = F.relu(margin - neg_dist).pow(2).mean()  # Negative pair distance
    # hinge_contrastive_loss = positive_loss + negative_loss

    # --- Combine all the losses ---
    # combined_loss = ce_loss + triplet_loss + hinge_contrastive_loss
    # combined_loss = ce_loss + triplet_loss
    combined_loss = triplet_loss

    return combined_loss


def train(model, train_loader, test_loader, vali_loader, num_epochs, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()

    for param in model.parameters():
        param.requires_grad = True

    for param in model.gpt_model.wte.parameters():
        param.requires_grad = False
    for param in model.gpt_model.wpe.parameters():
        param.requires_grad = False

    for layer in model.gpt_model.h:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.gpt_model.ln_f.parameters():
        param.requires_grad = True

    for param in model.bert_model.parameters():
        param.requires_grad = False

    # last_layer = model.bert_model.encoder.layer[-1]  # Access the last encoder layer
    # for param in last_layer.output.parameters():
    #     param.requires_grad = True
    for param in model.moment_model.parameters():
        param.requires_grad = False

    for param in model.bert_model.pooler.parameters():
        param.requires_grad = True

    for param in model.bert_model.pooler.parameters():
        param.requires_grad = True

    # for param in model.moment_model.encoder.block[-1].parameters():
    #     param.requires_grad = True



    # print(model.moment_model.encoder.block[-1])

    # for param in model.moment_model.parameters():
    #     param.requires_grad = True

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    ## tyoe 1-5
    optimizer = optim.Adam([
        {'params': model.gpt_model.h.parameters(), 'lr': 1e-5},
        {'params': model.gpt_model.ln_f.parameters(), 'lr': 1e-5},
        {'params': model.fuse_project.parameters(), 'lr': 5e-7},
        {'params': model.bert_linear.parameters(), 'lr': 5e-7},
        {'params': model.moment_model.parameters(), 'lr': 1e-6},
        {'params': model.bert_model.parameters(), 'lr': 5e-7},
        {'params': model.embed_project.parameters(), 'lr': 1e-5},
        {'params': model.output_project.parameters(), 'lr': 1e-5},
        {'params': model.residual_project.parameters(), 'lr': 1e-5},
    ])

    ## type 6
    # optimizer = optim.Adam([
    #     {'params': model.gpt_model.h.parameters(), 'lr': 1e-5},
    #     {'params': model.gpt_model.ln_f.parameters(), 'lr': 1e-5},
    #     {'params': model.fuse_project.parameters(), 'lr': 1e-7},
    #     {'params': model.bert_linear.parameters(), 'lr': 1e-7},
    #     {'params': model.moment_model.parameters(), 'lr': 1e-6},
    #     {'params': model.bert_model.parameters(), 'lr': 1e-7},
    #     {'params': model.embed_project.parameters(), 'lr': 1e-5},
    #     {'params': model.output_project.parameters(), 'lr': 1e-5},
    #     {'params': model.residual_project.parameters(), 'lr': 1e-5},
    # ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    log_file = ""

    best_loss = float('inf')
    for epoch in range(num_epochs):
        loss = 0

        # # Freeze all parameters initially
        # for param in model.bert_model.parameters():
        #     param.requires_grad = False
        # for param in model.bert_model.pooler.parameters():
        #     param.requires_grad = True
        # last_layer = model.bert_model.encoder.layer[-1]  # Access the last encoder layer
        # for param in last_layer.output.parameters():
        #     param.requires_grad = True

        for batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale in train_loader:
            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))

            output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings, batch_sent_report_embeddings, batch_negative_type_report_embeddings = model.predict_batch_contrastive(
                batch_text, batch_sent_reports, batch_negative_type_reprots,
                batch_seq_scale)

            # print(
            #     f"CHECKING train {batch_fusion_embeddings.shape}, {batch_sent_report_embeddings.shape}, {batch_negative_type_report_embeddings.shape}")
            optimizer.zero_grad()

            # Objective 1: MSE Loss
            mse_loss = criterion_mse(batch_pred_scale.float().to(torch.device('cuda')), output.float())

            # Objective 2: MAE Loss
            mae_loss = criterion_mae(batch_pred_scale.float().to(torch.device('cuda')), output.float())

            # Objective 3: Contrastive Loss
            # contrastive_loss = contrastive_loss_objective(batch_fusion_embeddings, batch_sent_report_embeddings,
            #                                               batch_negative_type_report_embeddings)

            ## Full Model
            contrastive_loss = contrastive_loss_objective(batch_series_embeddings, batch_sent_report_embeddings,
                                                          batch_negative_type_report_embeddings)
            # #
            combined_loss = mse_loss + mae_loss + contrastive_loss
            # combined_loss = mse_loss + mae_loss

            ## Ablation Model (No Textual Modality)
            # combined_loss = mse_loss + mae_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            combined_loss.backward(retain_graph=True)
            optimizer.step()
            # Apply gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            ## check gradients
            # total_norm = 0.0
            # for param in model.parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data.norm(2)  # L2 norm
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Gradient Norm: {total_norm}")

            loss += combined_loss.item()
            print(f'Step in Epoch: {epoch + 1}/{num_epochs}, loss: {combined_loss}')
            log_file += f'Step in Epoch: {epoch + 1}/{num_epochs}, loss: {combined_loss}\n'

        ## TYPE 1
        # if epoch + 1 >= 5:
        #     scheduler.step()
        # if epoch + 1 >= 5:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #     for param in model.bert_model.parameters():
        #         param.requires_grad = False

        ## TYPE 2
        # if epoch + 1 >= 5:
        #     scheduler.step()
        # if epoch + 1 >= 5:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #         # Unfreeze only the parameters in the pooler
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     last_layer = model.bert_model.encoder.layer[-2]  # Access the last encoder layer
        #     for param in last_layer.output.parameters():
        #         param.requires_grad = True
        # if epoch + 1 >= 10:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = True
        # if epoch + 1 >= 15:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #     for param in model.bert_model.parameters():
        #         param.requires_grad = False

        # ## type 3
        # if epoch + 1 >= 5:
        #     scheduler.step()
        # if epoch + 1 >= 5:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #         # Unfreeze only the parameters in the pooler
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     last_layer = model.bert_model.encoder.layer[-2]  # Access the last encoder layer
        #     for param in last_layer.output.parameters():
        #         param.requires_grad = True
        # if epoch + 1 >= 10:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = True

        ## type 4
        # if epoch + 1 >= 5:
        #     scheduler.step()
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     last_layer = model.bert_model.encoder.layer[-1]  # Access the last encoder layer
        #     for param in last_layer.output.parameters():
        #         param.requires_grad = True
        # if epoch + 1 >= 5:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #     for param in model.bert_model.parameters():
        #         param.requires_grad = False

        ## type 5
        # if epoch + 1 >= 5:
        #     scheduler.step()
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     last_layer = model.bert_model.encoder.layer[-3]  # Access the last encoder layer
        #     for param in last_layer.output.parameters():
        #         param.requires_grad = True
        # if epoch + 1 >= 5:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #     for param in model.bert_model.parameters():
        #         param.requires_grad = False


        ## type 6
        # if epoch + 1 >= 10:
        #     scheduler.step()
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     last_layer = model.bert_model.encoder.layer[-1]  # Access the last encoder layer
        #     for param in last_layer.output.parameters():
        #         param.requires_grad = True
        #     # for param in last_layer.intermediate.parameters():
        #     #     param.requires_grad = True
        # if epoch + 1 >= 10:
        #     for param in model.moment_model.parameters():
        #         param.requires_grad = False
        #     for param in model.bert_model.parameters():
        #         param.requires_grad = False


        # type 7
        scheduler.step()
        # if epoch + 1 >= 5:
        #     scheduler.step()
        #     # for param in model.moment_model.parameters():
        #     #     param.requires_grad = True
        #     for param in model.bert_model.pooler.parameters():
        #         param.requires_grad = True
        #     model.moment_model.encoder.block[-1]



        print(f'Epoch: {epoch + 1}/{num_epochs}, loss: {loss / len(train_loader)}')
        log_file += f'Epoch: {epoch + 1}/{num_epochs}, loss: {loss / len(train_loader)}\n'

        vali_combine_loss, vali_mse_loss, vali_mae_loss, vali_contras_loss = test(model, vali_loader)
        test_combine_loss, test_mse_loss, test_mae_loss, test_contras_loss = test(model, test_loader)
        print(
            f'Vali Combined Loss: {vali_combine_loss}, Vali MSE Loss: {vali_mse_loss}, Vali MAE Loss: {vali_mae_loss}, Vali Contrastive Loss: {vali_contras_loss}\n'
            f'Test Combined loss: {test_combine_loss}, Test MSE Loss: {test_mse_loss}, Test MAE Loss: {test_mae_loss}, Test Contrastive Loss: {test_contras_loss}')
        log_file += (
            f'Vali Combined Loss: {vali_combine_loss}, Vali MSE Loss: {vali_mse_loss}, Vali MAE Loss: {vali_mae_loss}\n'
            f'Test Combined loss: {test_combine_loss}, Test MSE Loss: {test_mse_loss}, Test MAE Loss: {test_mae_loss}')

        if output_path is not None:
            model_output_path = output_path
            log_output_path = os.path.join(output_path, "log.txt")
        else:
            model_output_path = "./model_output"
            log_output_path = os.path.join("./model_output", "log.txt")

        save_directory = Path(model_output_path)
        save_directory.mkdir(parents=True, exist_ok=True)

        lastest_loss = float(vali_mse_loss)

        if lastest_loss < best_loss:
            best_loss = lastest_loss
            save_path = os.path.join(model_output_path, "best_model.pth")
            model.save_model_combined(save_path = save_path)
            print(
                f'Saving Best Model @ iter{epoch + 1}: Vali Combined Loss: {vali_combine_loss}, Vali MSE Loss: {vali_mse_loss}, Vali MAE Loss: {vali_mae_loss}\n'
                f'Test Combined loss: {test_combine_loss}, Test MSE Loss: {test_mse_loss}, Test MAE Loss: {test_mae_loss}')
            log_file += (
                f'Saving Best Model @ iter{epoch + 1}: Vali Combined Loss: {vali_combine_loss}, Vali MSE Loss: {vali_mse_loss}, Vali MAE Loss: {vali_mae_loss}\n '
                f'Test Combined loss: {test_combine_loss}, Test MSE Loss: {test_mse_loss}, Test MAE Loss: {test_mae_loss}')
        save_path = os.path.join(model_output_path, "lastest_model.pth")
        model.save_model_combined(save_path = save_path)
        # checkpoint = torch.load('./test_save/test')
        # self.load_state_dict(checkpoint)
        # exit()
        # save_pretrained(model_output_path, config=model.config)
        with open(log_output_path, 'w') as f:
            f.write(log_file)


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Initialize loss accumulators
    combined_loss_output = 0
    mse_loss_output = 0
    mae_loss_output = 0
    contrastive_loss_output = 0

    # Wrap the test loader with tqdm for progress bar
    with torch.no_grad():
        # Create a progress bar using tqdm
        for batch_idx, (batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):
            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))

            # Make predictions using the model
            output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings, batch_sent_report_embeddings, batch_negative_type_report_embeddings = model.predict_batch_contrastive(
                batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale)

            # Calculate losses
            mse_loss = criterion_mse(output, batch_pred_scale.to(device))
            mae_loss = criterion_mae(output, batch_pred_scale.to(device))

            # Full model combined loss
            contrastive_loss = contrastive_loss_objective(batch_series_embeddings, batch_sent_report_embeddings,
                                                          batch_negative_type_report_embeddings)
            combined_loss = mse_loss + mae_loss + contrastive_loss

            # Update accumulated loss values
            combined_loss_output += combined_loss.item()
            mse_loss_output += mse_loss.item()
            mae_loss_output += mae_loss.item()
            contrastive_loss_output += contrastive_loss.item()

        # Calculate average losses
        avg_combined_loss = combined_loss_output / len(test_loader)
        avg_mse_loss = mse_loss_output / len(test_loader)
        avg_mae_loss = mae_loss_output / len(test_loader)
        avg_contrastive_loss = contrastive_loss_output / len(test_loader)

    # Return the average losses
    return avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss

    # print(f'Test loss: {loss / len(test_loader)}',end='\t')


def test_save(model, test_loader, scaler, output_folder=None):
    if output_folder is None:
        output_folder = "multi-model-forecast/test_samples_analysis/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Initialize loss accumulators
    combined_loss_output = 0
    mse_loss_output = 0
    mae_loss_output = 0
    contrastive_loss_output = 0

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Wrap the test loader with tqdm for progress bar
    with torch.no_grad():
        for batch_idx, (
        batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale,
        batch_pred_scale) in enumerate(tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):
            batch_sent_reports = list(map(list, zip(*batch_sent_reports)))
            batch_negative_type_reprots = list(map(list, zip(*batch_negative_type_reprots)))

            # Make predictions using the model
            output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings, batch_sent_report_embeddings, batch_negative_type_report_embeddings = model.predict_batch_contrastive(
                batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq_scale)

            # Save each sample in the batch
            for sample_idx, (
            text, sent_reports, negative_reports, seq, pred, seq_scale, pred_scale, prediction) in enumerate(zip(
                    batch_text, batch_sent_reports, batch_negative_type_reprots, batch_seq, batch_pred, batch_seq_scale,
                    batch_pred_scale, output)):

                # Create a subfolder for the sample
                sample_folder = os.path.join(output_folder, f"batch_{batch_idx}_sample_{sample_idx}")
                os.makedirs(sample_folder, exist_ok=True)
                print("check sample folder", sample_folder)
                # Save the original text file
                text_path = os.path.join(sample_folder, "original_text.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)

                # Save sent_reports
                for i, report in enumerate(sent_reports):
                    sent_report_path = os.path.join(sample_folder, f"sent_report_{i}.txt")
                    with open(sent_report_path, "w", encoding="utf-8") as f:
                        f.write(report)

                # Save negative type reports
                for i, report in enumerate(negative_reports):
                    negative_report_path = os.path.join(sample_folder, f"negative_report_{i}.txt")
                    with open(negative_report_path, "w", encoding="utf-8") as f:
                        f.write(report)

                # Function to reshape and save data in vertical format
                def save_vertical_format(data, sample_folder, file_name):
                    """
                    Save data in vertical format with columns Open, High, Low, and Close.
                    """
                    # Check if data is a PyTorch tensor, and convert to NumPy array if necessary
                    if isinstance(data, torch.Tensor):
                        data_array = data.cpu().numpy()
                    elif isinstance(data, np.ndarray):
                        data_array = data
                    else:
                        raise ValueError("Input data must be either a PyTorch tensor or a NumPy array.")

                    # Reshape the data to rows of 4 (Open, High, Low, Close)
                    reshaped_data = data_array.reshape(-1, 4)

                    # Create a DataFrame with proper column names
                    df = pd.DataFrame(reshaped_data, columns=["Open", "High", "Low", "Close"])

                    # Save the DataFrame to a CSV file
                    output_path = os.path.join(sample_folder, file_name)
                    df.to_csv(output_path, index=False)


                seq_path = "original_seq.csv"
                save_vertical_format(seq, sample_folder, seq_path)

                # Save predicted time series
                pred_path = "original_pred.csv"
                save_vertical_format(pred, sample_folder, pred_path)

                # Save scaled time series data
                seq_scale_path = "scaled_seq.csv"
                save_vertical_format(seq_scale, sample_folder, seq_scale_path)

                # Save scaled predictions
                pred_scale_path = "scaled_pred.csv"
                save_vertical_format(pred_scale, sample_folder, pred_scale_path)

                # Save model predictions
                prediction_scale_path = "scaled_predictions.csv"
                save_vertical_format(prediction, sample_folder, prediction_scale_path)

                # Save model predictions converted
                prediction_converted_path = "converted_predictions.csv"
                converted_predictions = scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 4)).reshape(-1, 4)
                save_vertical_format(converted_predictions, sample_folder, prediction_converted_path)



            # Calculate losses
            mse_loss = criterion_mse(output, batch_pred_scale.to(device))
            mae_loss = criterion_mae(output, batch_pred_scale.to(device))

            # Full model combined loss
            contrastive_loss = contrastive_loss_objective(batch_series_embeddings, batch_sent_report_embeddings,
                                                          batch_negative_type_report_embeddings)
            combined_loss = mse_loss + mae_loss + contrastive_loss

            # Update accumulated loss values
            combined_loss_output += combined_loss.item()
            mse_loss_output += mse_loss.item()
            mae_loss_output += mae_loss.item()
            contrastive_loss_output += contrastive_loss.item()

            print(f"Testing completed. Results saved in {output_folder}")


        # Calculate average losses
        avg_combined_loss = combined_loss_output / len(test_loader)
        avg_mse_loss = mse_loss_output / len(test_loader)
        avg_mae_loss = mae_loss_output / len(test_loader)
        avg_contrastive_loss = contrastive_loss_output / len(test_loader)

    # Return the average losses
    return avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss

def real_test(model, test_loader, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Initialize loss accumulators
    combined_loss_output = 0
    mse_loss_output = 0
    mae_loss_output = 0
    contrastive_loss_output = 0

    # Wrap the test loader with tqdm for progress bar
    with torch.no_grad():
        # Create a progress bar using tqdm
        for batch_idx, (batch_text, batch_seq, batch_pred, batch_seq_scale, batch_pred_scale) in enumerate(tqdm(test_loader, desc="Testing", ncols=100, dynamic_ncols=True)):

            # Make predictions using the model
            output, batch_fusion_embeddings, batch_series_embeddings, batch_text_embeddings = model.predict_single_case(batch_text, batch_seq_scale)

            # Calculate losses
            mse_loss = criterion_mse(batch_pred_scale.to(device), output)
            mae_loss = criterion_mae(batch_pred_scale.to(device), output)

            combined_loss = mse_loss + mae_loss

            # Update accumulated loss values
            combined_loss_output += combined_loss.item()
            mse_loss_output += mse_loss.item()
            mae_loss_output += mae_loss.item()

        # Calculate average losses
        avg_combined_loss = combined_loss_output / len(test_loader)
        avg_mse_loss = mse_loss_output / len(test_loader)
        avg_mae_loss = mae_loss_output / len(test_loader)
        avg_contrastive_loss = contrastive_loss_output / len(test_loader)

        denormalized_prediction = scaler.inverse_transform(output.cpu().numpy().reshape(-1,4))
        denormalized_truth = scaler.inverse_transform(batch_pred_scale.cpu().numpy().reshape(-1,4))


        correct_up_predictions = 0
        correct_down_predictions = 0
        total_up_predictions = 0
        total_down_predictions = 0
        cumulative_profit = 0

        # Iterate over each row in the prediction and truth tensors
        for i in range(len(denormalized_prediction)):
            # Extract Open, High, Low, Close values for both predicted and real data
            predicted_open, predicted_high, predicted_low, predicted_close = denormalized_prediction[i]
            real_open, real_high, real_low, real_close = denormalized_truth[i]

            # Determine the predicted and real direction (up or down) based on Close vs Open
            predicted_direction = 1 if predicted_close > predicted_open else -1
            real_direction = 1 if real_close > real_open else -1

            # Cumulative profit for correct predictions (profit when predicted direction is correct)
            if predicted_direction == real_direction:
                if predicted_direction == 1:  # Predicted up
                    correct_up_predictions += 1
                    total_up_predictions += 1
                    cumulative_profit += (real_close - real_open)
                elif predicted_direction == -1:  # Predicted down
                    correct_down_predictions += 1
                    total_down_predictions += 1
                    cumulative_profit += (real_open - real_close)
            elif real_direction == 1:  # Actual up but predicted down
                total_up_predictions += 1
            elif real_direction == -1:  # Actual down but predicted up
                total_down_predictions += 1

        # Calculate the hit rate for both upward and downward movements
        hit_rate_up = (correct_up_predictions / total_up_predictions) * 100 if total_up_predictions > 0 else 0
        hit_rate_down = (correct_down_predictions / total_down_predictions) * 100 if total_down_predictions > 0 else 0

        # Calculate the total hit rate
        total_hit_rate = ((correct_up_predictions + correct_down_predictions) / (
                    total_up_predictions + total_down_predictions)) * 100 if (
                                                                                         total_up_predictions + total_down_predictions) > 0 else 0

        # Print the results
        print(f"Hit Rate for Upward Movements: {hit_rate_up:.2f}%")
        print(f"Hit Rate for Downward Movements: {hit_rate_down:.2f}%")
        print(f"Total Hit Rate: {total_hit_rate:.2f}%")
        print(f"Cumulative Profit: {cumulative_profit:.2f}")

    # Return the average losses
    return avg_combined_loss, avg_mse_loss, avg_mae_loss, avg_contrastive_loss