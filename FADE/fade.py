import json
from pathlib import Path
from typing import List, Literal, Dict, Tuple
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import PIL

from FADE.datasets.utils import min_max_normalization
from FADE.utils.anomaly_detection import predict_classification, predict_segmentation
from FADE.utils.embeddings import extract_image_embeddings, extract_text_embeddings_by_batch
from FADE.utils.text_model import build_text_model, get_text_model
from FADE.datasets.base import IMAGENET_MEAN, IMAGENET_STD


class FADETextModel:
    def __init__(
        self,
        gem_model: torch.nn.Module,
        text_model_type: Literal["average", "softmax", "max", "lr", "mlp", "knn", "rf", "xgboost", "gmm"],
        device: Literal['cpu', 'cuda'] = 'cpu',
        mode: Literal['both', 'segmentation', 'classification'] = 'both'
    ):
        self.text_model_type = text_model_type
        self.device = device
        self.mode = mode

        self.gem_model = gem_model

        self.text_model = get_text_model(model_type=self.text_model_type)

    def fit(
        self,
        normal_texts: List[str],
        abnormal_texts: List[str],
        batch_size: int = 64,
    ):
        text_embeddings, text_labels = self.get_text_embeddings(
            normal_texts=normal_texts,
            abnormal_texts=abnormal_texts,
            batch_size=batch_size
        )
        self.text_model.fit(text_embeddings, text_labels)

    def get_text_embeddings(
        self,
        normal_texts: List[str],
        abnormal_texts: List[str],
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            normal_text_embeddings = extract_text_embeddings_by_batch(
                self.gem_model, normal_texts, batch_size=batch_size
            )
            abnormal_text_embeddings = extract_text_embeddings_by_batch(
                self.gem_model, abnormal_texts, batch_size=batch_size
            )
            text_embeddings = np.array(
                normal_text_embeddings + abnormal_text_embeddings)
            text_labels = np.array(
                [0] * len(normal_text_embeddings) + [1] *
                len(abnormal_text_embeddings)
            )
            return text_embeddings, text_labels

    @staticmethod
    def get_prompts(prompt_paths: List[Path | str], classname: str) -> Tuple[List[str], List[str]]:
        all_normal_texts, all_abnormal_texts = [], []
        for prompt_path in prompt_paths:
            with open(prompt_path) as fp:
                prompts = json.load(fp)

            normal_text = [
                t.format(classname=classname) for t in prompts["normal"]["prompts"]
            ]
            abnormal_text = [
                t.format(classname=classname) for t in prompts["abnormal"]["prompts"]
            ]
            all_normal_texts.extend(normal_text)
            all_abnormal_texts.extend(abnormal_text)
        return all_normal_texts, all_abnormal_texts


class FADEImageModel:
    def __init__(
        self,
        gem_model: torch.nn.Module,
        classification_img_size: int,
        segmentation_img_sizes: List[int],
        square: bool = False,
        device: Literal['cpu', 'cuda'] = 'cpu',
        mode: Literal['both', 'segmentation', 'classification'] = 'both'
    ):
        self.device = device
        self.mode = mode
        self.classification_img_size = classification_img_size
        self.segmentation_img_sizes = segmentation_img_sizes

        img_sizes = list(
            {classification_img_size, *segmentation_img_sizes}
        )

        self.transform_img = {}
        # Multiple resize transforms
        for sz in img_sizes:
            transform_img = [
                transforms.Resize(
                    (sz, sz) if square else sz,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                # transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img[sz] = transforms.Compose(transform_img)

        self.gem_model = gem_model

    def extract_image_embeddings(
        self,
        image: PIL.Image,
    ) -> Dict[int, np.ndarray]:

        multiscale_images = {}

        for sz, transform in self.transform_img.items():
            multiscale_images[sz] = transform(image)

        image_embeddings = extract_image_embeddings(
            multiscale_images=multiscale_images,
            gem_model=self.gem_model,
            device=self.device
        )
        return image_embeddings


class FADE:
    def __init__(
        self,
        gem_model: torch.nn.Module,
        classification_mode: Literal["none",
                                     "language", "vision", "both"] = "both",
        segmentation_mode: Literal["none",
                                   "language", "vision", "both"] = "both",
        language_classification_feature: Literal['clip', 'gem'] = 'clip',
        language_segmentation_feature: Literal['clip', 'gem'] = 'gem',
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        self.gem_model = gem_model

        self.classification_mode = classification_mode
        self.segmentation_mode = segmentation_mode

        self.classification_text_model = None
        self.segmentation_text_model = None

        self.language_classification_feature = language_classification_feature
        self.language_segmentation_feature = language_segmentation_feature
        self.device = device

        self.image_model = None

    def build_model(self):
        self.image_model = FADEImageModel(
            gem_model=self.gem_model,
            classification_img_size=384,
            segmentation_img_sizes=[384, 512],
            device=self.device,
        )

        if self.classification_mode == "language" or self.classification_mode == "both":
            self.classification_text_model = FADETextModel(
                gem_model=self.gem_model,
                text_model_type="average",
                device=self.device,
            )

        if self.segmentation_mode == "language" or self.segmentation_mode == "both":
            self.segmentation_text_model = FADETextModel(
                gem_model=self.gem_model,
                text_model_type="average",
                device=self.device,
            )

    # TODO implement vision guided
    def predict(self, image: PIL.Image) -> Dict[int, float]:
        image_embeddings = self.image_model.extract_image_embeddings(image)

        if self.classification_mode != "vision" and self.segmentation_mode != "vision":
            language_guided_scores = predict_classification(
                text_model=self.classification_text_model,
                image_embeddings=image_embeddings,
                img_size=self.image_model.classification_img_size,
                feature_type=self.language_classification_feature,
            )

            language_guided_maps = predict_segmentation(
                model=self.segmentation_text_model,
                image_embeddings=image_embeddings,
                img_sizes=self.image_model.segmentation_img_sizes,
                feature_type=self.language_segmentation_feature,
                patch_size=self.gem_model.model.visual.patch_size,
                segmentation_mode="language",
            )

        scores = np.clip(language_guided_scores, 0, 1)

        segmentations = min_max_normalization(language_guided_maps)
        segmentations = np.clip(segmentations, 0, 1)
        segmentations = (segmentations * 255).astype("uint8")

        return scores, segmentations
