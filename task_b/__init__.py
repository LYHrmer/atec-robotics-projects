"""Task B (ATEC-TaskB-B2wPiper) mobility bootstrap.

`control` holds open-loop policies that read only public proprioception and a
static joint/action schema. `evaluate` runs one fresh-process episode of the
original task and writes auditable artifacts. Neither module changes the
official physics, assets, actions, rewards or terminations.

`evaluate` must be started before Isaac Sim exists, so nothing is imported here.
"""
