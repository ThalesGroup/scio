This branch exposes a trained [Tiniest](https://github.com/xvel/cifar10-tiniest) model to `torch.hub` for tutorials.

# Loading the trained model

```python
import torch
torch.hub.load("ThalesGroup/scio:hub", "tiniest", trust_repo=True, verbose=False)
```

# Model Specifications

This is a classification model trained on CIFAR10, specifically on the `train/` split from [this](https://huggingface.co/datasets/ego-thales/cifar10) dataset.

### Performance

It achieves **94.6% accuracy** on the `test/` set (98.7% on `train/`) despite having **only 97,530 trainable parameters**! Here is a full classification report generated with `sklearn.metrics.classification_report`:

```
              precision    recall  f1-score   support

    airplane      0.954     0.947     0.950      1000
  automobile      0.950     0.984     0.967      1000
        bird      0.929     0.933     0.931      1000
         cat      0.891     0.877     0.884      1000
        deer      0.960     0.951     0.955      1000
         dog      0.913     0.910     0.911      1000
        frog      0.964     0.973     0.969      1000
       horse      0.966     0.963     0.964      1000
        ship      0.966     0.972     0.969      1000
       truck      0.969     0.953     0.961      1000

    accuracy                          0.946     10000
   macro avg      0.946     0.946     0.946     10000
weighted avg      0.946     0.946     0.946     10000
```

### Architecture & training procedure

The model and training procedure were taken *without modification* (except using our `train/` split and `num_epochs = 300`) from [github.com/xvel/cifar10-tiniest](https://github.com/xvel/cifar10-tiniest). Its author reportedly got inspired by [github.com/soyflourbread/cifar10-tiny](https://github.com/soyflourbread/cifar10-tiny).

<details>
  <summary>Click to <strong>view the layers</strong>.</summary>

  Below is a *reformatted* layerwise overview of the model, generated with [torchinfo](https://github.com/tyleryep/torchinfo):

  ```
  =========================================================================================
  Layer (type:depth-idx)        Input Shape        Output Shape        Param #    Param %
  =========================================================================================
  Tiniest                       [*, 3, 32, 32]     [*, 10]             --              --
  ├─Conv2d: 1-1                 [*, 3, 32, 32]     [*, 48, 32, 32]     1,344        1.38%
  ├─LayerNorm2d: 1-2            [*, 48, 32, 32]    [*, 48, 32, 32]     --              --
  │    └─LayerNorm: 2-1         [*, 32, 32, 48]    [*, 32, 32, 48]     96           0.10%
  ├─Block: 1-3                  [*, 48, 32, 32]    [*, 48, 32, 32]     48           0.05%
  │    └─Conv2d: 2-2            [*, 12, 32, 32]    [*, 12, 32, 32]     120          0.12%
  │    └─Conv2d: 2-3            [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─AvgPool2d: 2-4         [*, 12, 32, 32]    [*, 12, 32, 32]     --              --
  │    └─Conv2d: 2-5            [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─LayerNorm2d: 2-6       [*, 48, 32, 32]    [*, 48, 32, 32]     --              --
  │    │    └─LayerNorm: 3-1    [*, 32, 32, 48]    [*, 32, 32, 48]     96           0.10%
  │    └─Conv2d: 2-7            [*, 48, 32, 32]    [*, 96, 32, 32]     4,704        4.82%
  │    └─Conv2d: 2-8            [*, 48, 32, 32]    [*, 48, 32, 32]     2,352        2.41%
  ├─Block: 1-4                  [*, 48, 32, 32]    [*, 48, 32, 32]     48           0.05%
  │    └─Conv2d: 2-9            [*, 12, 32, 32]    [*, 12, 32, 32]     120          0.12%
  │    └─Conv2d: 2-10           [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─AvgPool2d: 2-11        [*, 12, 32, 32]    [*, 12, 32, 32]     --              --
  │    └─Conv2d: 2-12           [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─LayerNorm2d: 2-13      [*, 48, 32, 32]    [*, 48, 32, 32]     --              --
  │    │    └─LayerNorm: 3-2    [*, 32, 32, 48]    [*, 32, 32, 48]     96           0.10%
  │    └─Conv2d: 2-14           [*, 48, 32, 32]    [*, 96, 32, 32]     4,704        4.82%
  │    └─Conv2d: 2-15           [*, 48, 32, 32]    [*, 48, 32, 32]     2,352        2.41%
  ├─Block: 1-5                  [*, 48, 32, 32]    [*, 48, 32, 32]     48           0.05%
  │    └─Conv2d: 2-16           [*, 12, 32, 32]    [*, 12, 32, 32]     120          0.12%
  │    └─Conv2d: 2-17           [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─AvgPool2d: 2-18        [*, 12, 32, 32]    [*, 12, 32, 32]     --              --
  │    └─Conv2d: 2-19           [*, 12, 32, 32]    [*, 12, 32, 32]     600          0.62%
  │    └─LayerNorm2d: 2-20      [*, 48, 32, 32]    [*, 48, 32, 32]     --              --
  │    │    └─LayerNorm: 3-3    [*, 32, 32, 48]    [*, 32, 32, 48]     96           0.10%
  │    └─Conv2d: 2-21           [*, 48, 32, 32]    [*, 96, 32, 32]     4,704        4.82%
  │    └─Conv2d: 2-22           [*, 48, 32, 32]    [*, 48, 32, 32]     2,352        2.41%
  ├─Conv2d: 1-6                 [*, 48, 32, 32]    [*, 80, 16, 16]     3,920        4.02%
  ├─LayerNorm2d: 1-7            [*, 80, 16, 16]    [*, 80, 16, 16]     --              --
  │    └─LayerNorm: 2-23        [*, 16, 16, 80]    [*, 16, 16, 80]     160          0.16%
  ├─Block: 1-8                  [*, 80, 16, 16]    [*, 80, 16, 16]     80           0.08%
  │    └─Conv2d: 2-24           [*, 20, 16, 16]    [*, 20, 16, 16]     200          0.21%
  │    └─Conv2d: 2-25           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─AvgPool2d: 2-26        [*, 20, 16, 16]    [*, 20, 16, 16]     --              --
  │    └─Conv2d: 2-27           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─LayerNorm2d: 2-28      [*, 80, 16, 16]    [*, 80, 16, 16]     --              --
  │    │    └─LayerNorm: 3-4    [*, 16, 16, 80]    [*, 16, 16, 80]     160          0.16%
  │    └─Conv2d: 2-29           [*, 80, 16, 16]    [*, 160, 16, 16]    12,960      13.29%
  │    └─Conv2d: 2-30           [*, 80, 16, 16]    [*, 80, 16, 16]     6,480        6.64%
  ├─Block: 1-9                  [*, 80, 16, 16]    [*, 80, 16, 16]     80           0.08%
  │    └─Conv2d: 2-31           [*, 20, 16, 16]    [*, 20, 16, 16]     200          0.21%
  │    └─Conv2d: 2-32           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─AvgPool2d: 2-33        [*, 20, 16, 16]    [*, 20, 16, 16]     --              --
  │    └─Conv2d: 2-34           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─LayerNorm2d: 2-35      [*, 80, 16, 16]    [*, 80, 16, 16]     --              --
  │    │    └─LayerNorm: 3-5    [*, 16, 16, 80]    [*, 16, 16, 80]     160          0.16%
  │    └─Conv2d: 2-36           [*, 80, 16, 16]    [*, 160, 16, 16]    12,960      13.29%
  │    └─Conv2d: 2-37           [*, 80, 16, 16]    [*, 80, 16, 16]     6,480        6.64%
  ├─Block: 1-10                 [*, 80, 16, 16]    [*, 80, 16, 16]     80           0.08%
  │    └─Conv2d: 2-38           [*, 20, 16, 16]    [*, 20, 16, 16]     200          0.21%
  │    └─Conv2d: 2-39           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─AvgPool2d: 2-40        [*, 20, 16, 16]    [*, 20, 16, 16]     --              --
  │    └─Conv2d: 2-41           [*, 20, 16, 16]    [*, 20, 16, 16]     1,000        1.03%
  │    └─LayerNorm2d: 2-42      [*, 80, 16, 16]    [*, 80, 16, 16]     --              --
  │    │    └─LayerNorm: 3-6    [*, 16, 16, 80]    [*, 16, 16, 80]     160          0.16%
  │    └─Conv2d: 2-43           [*, 80, 16, 16]    [*, 160, 16, 16]    12,960      13.29%
  │    └─Conv2d: 2-44           [*, 80, 16, 16]    [*, 80, 16, 16]     6,480        6.64%
  ├─AdaptiveAvgPool2d: 1-11     [*, 80, 16, 16]    [*, 80, 1, 1]       --              --
  ├─Linear: 1-12                [*, 80]            [*, 10]             810          0.83%
  =========================================================================================
  Total params: 97,530
  Trainable params: 97,530
  Non-trainable params: 0
  Total mult-adds (Units.MEGABYTES): 44.73
  =========================================================================================
  Input size (MB): 0.01
  Forward/backward pass size (MB): 9.05
  Params size (MB): 0.39
  Estimated Total Size (MB): 9.45
  =========================================================================================
  ```

</details>

<br/>
