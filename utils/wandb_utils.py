import os
from typing import Optional

import wandb


def init_wandb(
    project: str,
    run_name: str,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    # - Initialize wandb run with given project/config
    # - Skip init if already active
    if wandb.run is not None:
        return wandb.run

    run = wandb.init(
        project=project,
        name=run_name,
        config=config or {},
        entity=entity,
        reinit=True,
    )
    return run
