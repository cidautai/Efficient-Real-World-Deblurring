## How to add a new loss into your training schedule

As in the other submodules, the `__init__.py` is the file that connects the code in `/losses` folder with the training schedule. All the relevant functions must be imported in the `__init__.py` from `loss.py` (or any other script that you may define in this subfolder). 

You may follow the examples in the `loss.py` script (such as `PixelLoss` or `PerceptualLoss`) to define a new loss. This must be a class inherited by `nn.Module`:

```python
class NewLoss(nn.Module):
    def __init__(self, loss_weight, *args):
        super(NewLoss, self).__init__()
        self.loss_weight = loss_weight
        ...
        self.loss = ...

    def forward(self, pred, target, *args):
        ...
        return self.loss_weight * self.loss(pred, target, *args)
```

> Is recommended to add a `loss_weight` arg to the definition of the new loss, as it helps to control the weighting of all the losses. Also, you may define a `loss_criterion` to stablish which is the supervised distance defined between the predicted and target tensors (e.g. l1, l2 or charbonnier). You can check this definition in the other example losses.

The next step in the addition of a loss is to import it in the `__init__.py` script. You have to add it to the if/else statements in the `create_loss` and `calculate_loss` functions. You may follow how this is done for the other losses. 

Finally, you need to add to your configuration file the keys related to this new loss. In the `train` key in this file you may add:

```yaml
train:
    ...
    new_loss_flag: True
    new_loss_criterion: l1
    new_loss_reduction: mean
    new_loss_weight: [0.1, 0.1, ...]
    ...
```

> If you want to use the new loss, check that the flag is set to True in `new_loss_flag` key.
