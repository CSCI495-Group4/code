"""
Model A: ResNet-18 + GRU fusion for multimodal emotion recognition.

- Image path:   100x100 grayscale facial images (Balanced RAF-DB)
                → resized & normalized → ResNet-18 backbone → 512-D vector
- Text path:    Short text snippets (emotions.csv)
                → token ids → Embedding → GRU → 512-D vector
- Fusion:       Concatenate [512-D image, 512-D text] → 1024-D
                → Dropout(0.5) → Linear → 7 emotion logits

This file only defines the model; training & evaluation live in train_eval_modelA.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ModelA_MultimodalEmotionNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        num_classes: int = 7,
    ) -> None:
        """
        Args:
            vocab_size: size of the text vocabulary
            embed_dim: dimensionality of word embeddings
            hidden_dim: hidden size for GRU and projected image features
            num_classes: number of emotion classes (7 for Balanced RAF-DB)
        """
        super().__init__()

        # ---- Image encoder (ResNet-18, pretrained on ImageNet) -----------------
        base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # remove final FC layer -> keep conv + pooling
        self.image_encoder = nn.Sequential(*list(base_resnet.children())[:-1])
        # project 512-D ResNet output to hidden_dim
        self.image_fc = nn.Linear(512, hidden_dim)

        # ---- Text encoder (Embedding + GRU) ------------------------------------
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ---- Fusion + classification head -------------------------------------
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images:  (B, 3, H, W) image batch
            text_ids: (B, T) padded token id sequences

        Returns:
            logits: (B, num_classes) unnormalized class scores
        """

        # ----- Image branch -----------------------------------------------------
        # ResNet-18 outputs (B, 512, 1, 1) → flatten to (B, 512)
        img_feat = self.image_encoder(images).view(images.size(0), -1)
        img_feat = self.image_fc(img_feat)

        # ----- Text branch ------------------------------------------------------
        embedded = self.embedding(text_ids)          # (B, T, embed_dim)
        _, h_n = self.gru(embedded)                  # h_n: (1, B, hidden_dim)
        text_feat = h_n.squeeze(0)                   # (B, hidden_dim)

        # ----- Fusion & classification -----------------------------------------
        fused = torch.cat([img_feat, text_feat], dim=1)  # (B, hidden_dim * 2)
        fused = self.dropout(fused)
        logits = self.fc(fused)
        return logits
