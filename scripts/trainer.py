import os
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from utils.data_preprocessing import collate_fn


def create_checkpoint_folder(base_path='checkpoints', log=False):
    setting_number = 1

    while True:
        if not log:
            folder_name = f'setting_#{setting_number}'
        else:
            folder_name = f'setting_#{setting_number}/logs'
        path = os.path.join(base_path, folder_name)

        if not os.path.exists(path):
            os.makedirs(path)

            return path

        setting_number += 1


class Trainer:
    def __init__(self, **kwargs):
        self.train_loss = None
        self.kwargs = kwargs

        self.q_model = kwargs.get('q_model')
        self.p_model = kwargs.get('p_model')
        self.optimizer = kwargs.get('optimizer')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.lr = kwargs.get('lr', 1e-4)
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 100)
        self.num_patience = kwargs.get('num_patience', 5)

        self.criterion = kwargs.get('criterion')
        self.scheduler = kwargs.get('scheduler', None)

        self.train_data = kwargs.get('train_data')
        self.val_data = kwargs.get('val_data')
        self.test_data = kwargs.get('test_data')

        self.seed = kwargs.get('seed', 42)
        self.checkpoint_dir = create_checkpoint_folder(base_path=kwargs.get('checkpoint_dir', './checkpoints'))
        self.log_dir = create_checkpoint_folder(base_path=kwargs.get('checkpoint_dir', './checkpoints'), log=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self._set_seed()
        self._init_data()

        self.q_model.to(self.device)
        self.p_model.to(self.device)

    @staticmethod
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def _init_data(self):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    def train(self):
        os.makedirs(self.checkpoint_dir + '/ckpt', exist_ok=True)
        max_top1 = 0.0
        patience = 0

        epoch_losses = list()

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.num_epochs

        pbar = tqdm(total=total_steps, unit="step", dynamic_ncols=True)

        for epoch in range(self.num_epochs):
            self.q_model.train()
            self.p_model.train()

            epoch_loss = self.train_step(epoch, pbar)
            epoch_losses.append(epoch_loss)

            val_top1, val_top5, val_top10 = self.evaluate(self.val_loader)
            self.writer.add_scalar('Val/Top1', val_top1, epoch)
            self.writer.add_scalar('Val/Top5', val_top5, epoch)
            self.writer.add_scalar('Val/Top10', val_top10, epoch)

            tqdm.write(f"{{'epoch': {epoch}, 'val_top1': {val_top1:.4f}, 'val_top5': {val_top5:.4f}, 'val_top10': {val_top10:.4f}}}")

            if self.num_patience > 0:
                if val_top1 > max_top1:
                    max_top1 = val_top1

                    torch.save(self.q_model.state_dict(), os.path.join(self.checkpoint_dir + '/ckpt', 'best_query_model.pth'))
                    torch.save(self.p_model.state_dict(), os.path.join(self.checkpoint_dir + '/ckpt', 'best_passage_model.pth'))
                
                    patience = 0
                else:
                    patience += 1

                if patience >= self.num_patience:
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break

            torch.save(self.q_model.state_dict(), os.path.join(self.checkpoint_dir + '/ckpt', 'last_query_model.pth'))
            torch.save(self.p_model.state_dict(), os.path.join(self.checkpoint_dir + '/ckpt', 'last_passage_model.pth'))

        pbar.close()

    def train_step(self, epoch, pbar):
        total_loss = 0.0
        total_steps = len(self.train_loader)

        for i, (queries, passges, query_len) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            B, N, _, _, _ = queries.shape
            random_indices = torch.randint(0, N, (B,))
            random_indices = torch.min(random_indices, query_len - 1)

            batch_indices = torch.arange(B)
            queries = queries[batch_indices, random_indices]

            queries = queries.to(self.device)
            passges = passges.to(self.device)

            q_emb = F.normalize(self.q_model(queries), p=2, dim=-1)
            p_emb = F.normalize(self.p_model(passges), p=2, dim=-1)

            sim_scores = torch.matmul(q_emb, p_emb.T)
            targets = torch.arange(0, self.batch_size).long().to(self.device)

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.q_model.zero_grad()
            self.p_model.zero_grad()

            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"})
            pbar.update(1)

            global_step = epoch * total_steps + i

            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Learning Rate', current_lr, global_step)

        avg_loss = total_loss / total_steps

        return avg_loss

    def evaluate(self, data_loader):
        p_embs = list()

        top_k_values = [1, 5, 10]
        total_correct = {k: 0 for k in top_k_values}
        total_samples = 0

        self.p_model.eval()
        with torch.no_grad():
            for _, passages, _ in data_loader:
                passages = passages.to(self.device)
                p_emb = F.normalize(self.p_model(passages), p=2, dim=-1)
                p_embs.append(p_emb)

        p_embs = torch.cat(p_embs, dim=0)

        self.q_model.eval()
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc="Evaluation", leave=False, unit='step', dynamic_ncols=True) as pbar:
                current_offset = 0
                for queries, _, query_len in data_loader:
                    B, N, _, _, _ = queries.shape

                    random_indices = torch.randint(0, N, (B,))
                    random_indices = torch.min(random_indices, query_len - 1)

                    batch_indices = torch.arange(B)
                    queries = queries[batch_indices, random_indices]

                    queries = queries.to(self.device)

                    batch_size = queries.size(0)
                    ground_truth_indices = torch.arange(
                        current_offset, current_offset + batch_size
                    ).to(self.device)

                    q_emb = F.normalize(self.q_model(queries), p=2, dim=-1)
                    sim_scores = torch.matmul(q_emb, p_embs.T)

                    for k in top_k_values:
                        _, topk_indices = torch.topk(sim_scores, k=k, dim=-1)

                        correct = topk_indices.eq(ground_truth_indices.unsqueeze(1))
                        total_correct[k] += correct.sum().item()

                    total_samples += queries.size(0)
                    current_offset += batch_size

                    pbar.update(1)

        val_top1 = total_correct[1] / total_samples
        val_top5 = total_correct[5] / total_samples
        val_top10 = total_correct[10] / total_samples

        return val_top1, val_top5, val_top10


    def test(self):
        if os.path.exists(os.path.join(self.checkpoint_dir + '/ckpt', 'best_query_model.pth')):
            self.p_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir + '/ckpt', 'best_passage_model.pth')))
            self.q_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir + '/ckpt', 'best_query_model.pth')))

            best_test_top1, best_test_top5, best_test_top10 = self.evaluate(self.test_loader)

            print(f"Best Test Top1: {best_test_top1:.4f}")
            print(f"Best Test Top5: {best_test_top5:.4f}")
            print(f"Best Test Top10: {best_test_top10:.4f}")
        
        self.p_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir + '/ckpt', 'last_passage_model.pth'), weights_only=True))
        self.q_model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir + '/ckpt', 'last_query_model.pth'), weights_only=True))

        last_test_top1, last_test_top5, last_test_top10 = self.evaluate(self.test_loader)

        print(f"Last Test Top1: {last_test_top1:.4f}")
        print(f"Last Test Top5: {last_test_top5:.4f}")
        print(f"Last Test Top10: {last_test_top10:.4f}")
