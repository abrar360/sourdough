"""wandb wrapper so the training code path is identical with wandb on or off."""
try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    def __init__(self, cfg, extra_config: dict):
        self.run = None
        if cfg.wandb_mode != "disabled":
            if wandb is None:
                raise ImportError("wandb is not installed; pip install wandb or set wandb_mode=disabled")
            self.run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.name,
                group=cfg.wandb_group, tags=cfg.wandb_tags, mode=cfg.wandb_mode,
                config={**cfg.to_dict(), **extra_config},
            )
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")

    def log(self, data: dict):
        if self.run is not None:
            wandb.log(data)

    def image(self, key, fig, epoch):
        if self.run is not None:
            wandb.log({key: wandb.Image(fig), "epoch": epoch})

    def summary(self, data: dict):
        if self.run is not None:
            for k, v in data.items():
                wandb.run.summary[k] = v

    def finish(self):
        if self.run is not None:
            wandb.finish()
