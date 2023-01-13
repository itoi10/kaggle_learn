from __future__ import annotations
import json

import torch
from PIL import Image
from pytorch_pretrained_vit import ViT
from torchvision import transforms


def recognition(img_path: str) -> list[str]:
    """画像に写っているものが何かを推測する

    Parameters
    ----------
    img_path : str
        画像のパス

    Returns
    -------
    list[str]
        画像の推測結果上位３つのリスト

    Examples
    --------
    >>> print(recognition("./img/cat01.jpg"))
    ['Egyptian cat', 'tabby, tabby cat', 'tiger cat']

    """

    model = ViT("B_16_imagenet1k", pretrained=True)

    labels_map = json.load(open("labels_map.json"))
    labels_map = list(labels_map.values())

    img = Image.open(img_path)
    tfms = transforms.Compose(
        [
            transforms.Resize(model.image_size),  # サイズ変更 384x384
            transforms.ToTensor(),  # Tensor型にする
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 平均,標準偏差が0.5になるようにRBG変換
        ]
    )

    img = tfms(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img).squeeze(0)

    # pred = torch.argmax(outputs)
    # pred_label = labels_map[pred]
    # return pred_label

    _, preds = torch.topk(outputs, 3)

    return [labels_map[k] for k in preds]


if __name__ == "__main__":
    preds = recognition("./img/cat01.jpg")
    print(preds)
