## Waymax Dynamics

This module contains implementations of dynamics models available for use within
Waymax. A dynamics model computes the successor state (e.g. position, yaw,
velocity) of one or more objects given an action (e.g. steering and
acceleration) via the `forward` method.

The following dynamics models are available:

-   `InvertibleBicycleModel`. A kinematically realistic model using a 2D action
    (acceleration, steering curvature).

-   `DeltaLocal`. A position-based model using a 3D action (dx, dy, dyaw)
    representing the displacement of an object relative to the current position
    and orientation. This model does not check for infeasible actions, and large
    displacements may lead to unrealistic behavior.

-   `DeltaGlobal`. A position-based model using a 3D action (dx, dy, dyaw)
    representing the displacement of an object relative to the global coordinate
    system. This model does not check for infeasible actions, and large
    displacements may lead to unrealistic behavior.

-   `StateDynamics`. A position-based model using a 5D action (x, y, yaw,
    velocity x, velocity y) that directly sets values in the state in global
    coordinates. This model does not check for infeasible actions.

The `DiscreteActionSpaceWrapper` can be used to discretize any dynamics model
based on a user-specified number of bins for each action dimension.
