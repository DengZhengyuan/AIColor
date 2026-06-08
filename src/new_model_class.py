import torch
import torch.nn as nn


class BaseCoatingModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_layers,
        neurons_per_layer,
        learning_rate,
        dropout_rate,
        activation_fn,
        delta=0.5,
        loss_type="huber",
        weight_decay=0.0,
    ):
        super(BaseCoatingModel, self).__init__()
        if n_layers != len(neurons_per_layer):
            raise ValueError(
                f"n_layers={n_layers} but got {len(neurons_per_layer)} widths"
            )

        layers = []
        in_features = input_size
        for width in neurons_per_layer:
            layers.append(nn.Linear(in_features, width))
            layers.append(self._activation(activation_fn))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = width

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type

        if loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=delta)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.apply(lambda module: self._init_weights(module, activation_fn))

    @staticmethod
    def _activation(name):
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        if name == "elu":
            return nn.ELU()
        if name == "silu":
            return nn.SiLU()
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        raise ValueError(f"Unsupported activation_fn: {name}")

    @staticmethod
    def _init_weights(module, activation_fn):
        if not isinstance(module, nn.Linear):
            return

        if activation_fn in {"relu", "leaky_relu"}:
            nonlinearity = "leaky_relu" if activation_fn == "leaky_relu" else "relu"
            nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
        else:
            nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

    def lossfunction(self, output, target):
        return self.criterion(output, target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.65
        )
        return optimizer, scheduler


class Coating_train_2(BaseCoatingModel):
    pass
