# task binary
<details>
<summary>任务设置</summary>
二分类 仅1期数据
</details>

## model: inception_v3

    acc:0.947895 auc:0.958379
<details>
<summary>模型参数</summary>
pretrained inception

Note: there is a bug that the img was resize to 299,_ and crop the center 299 pixels which mean the edge will be missing

the norm setting is used the default
</details>

## model inception_v3
    acc:0.924568 auc:0.947926
<details>
<summary>模型参数</summary>
non_pretrained inception

Note: there is a bug that the img was resize to 299,_ and crop the center 299 pixels which mean the edge will be missing

the norm setting is used the default
</details>

## model Vess(LadderNet)+Class(inception_v3)

    Test: acc:0.877310 auc:0.904614
<details>
<summary>模型参数</summary>
setting: vess pretrained class pretrained

Note: there is a bug that the img was resize to 299,_ and crop the center 299 pixels which mean the edge will be missing(not sure)

the norm setting is used the default
</details>

# stage classification
<details>
<summary>任务设置</summary>
ROP分期 仅1期数据 舍弃未标注
</details>

## BaseLine inception_V3

    acc:0.966920 auc:0.987690
<details><summary>模型参数</summary>
pretrained inception_v3

the norm setting is used the default
</details>
    
    acc:0.971168 auc:0.993107
<details><summary>模型参数</summary>
pretrained inception_v3

```python
data_transform=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4623,0.3856,0.2822],
    std=[0.2527,0.1889,0.1334])
    # the mean and std is calculate by rop1 13 samples
    ])
```
</details>

## FR-UNet+inceptionV3 

    acc:0.933536 auc:0.984920 
<details>
<summary>模型参数</summary>

there is a bug that the vessel result was resize as 512,512 but crop to 300,300 which mean we will miss the right down information
```python
vessel_transform={
        transforms.Normalize([0.3968], [0.1980]),
        resize=(300,300)
    }
data_transform=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4623,0.3856,0.2822],
    std=[0.2527,0.1889,0.1334])
    # the mean and std is calculate by rop1 13 samples
    ])
```
</details>

    acc:0.969651 auc:0.996000

<details>
<summary>模型参数</summary>
Note we do not set the mannual seed so this reasearch can not reimplement perfectly
```python
# replace the blue channel(img[2]) with vessel segmentation result
train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            ])
vessel_transform={
        transforms.Normalize([0.3968], [0.1980]),
        resize=(300,300)
    }
data_transform=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4623,0.3856,0.2822],
    std=[0.2527,0.1889,0.1334])
    # the mean and std is calculate by rop1 13 samples
    ])
```
</details>
