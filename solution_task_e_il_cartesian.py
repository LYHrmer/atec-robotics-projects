"""Experimental learned Cartesian-goal servo residual over analytic DLS.

RGBD perception, IK path planning, the grasp state machine and the gripper
remain hand-coded; only the six-joint residual is learned.
"""
from tools.task_e.il.cartesian_policy import CartesianResidualSolution as AlgSolution
