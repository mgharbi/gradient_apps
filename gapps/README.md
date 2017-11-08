- Avoid implicit Vars for now.

List of ops with wrong/not implemented gradients:
+ BoundaryConditions::constant_exterior (fixed)
+ BoundaryConditions::repeat_edge (fixed)

Experimental notes while scheduling Bilateral Layer
- difficult to schedule a producer when multiple consumers
- in particular, no handle on the exact intermediate functions
- tricky to access the Rdoms and rfactors auto-generated
