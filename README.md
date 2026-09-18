Neuron Redundancy Biases ResNets Toward Collective Transport

This project studies why an overparameterized ResNet may learn a collective, class-level solution even when it has enough capacity to transport every training sample independently.

The central hypothesis is:

Entropic bias favors neuron redundancy. Solutions with functionally similar neurons occupy broader low-loss regions of parameter space.

Neuron redundancy produces collective transport. Redundant neurons implement shared class-level operations rather than controlling individual samples separately.

Collective transport may support generalization. Shared operations depend less strongly on the identities of particular training samples.

Setup

We interpret a depth-(L) ResNet as a discrete dynamical system,

[
h^{(\ell+1)}=h^{(\ell)}+\frac{1}{L}f_\ell!\left(h^{(\ell)}\right),
]

where depth plays the role of time and each residual block defines a velocity field in representation space.

For a two-class dataset, the inputs are embedded in three dimensions and the two classes are assigned targets above and below the original data plane. This makes the hidden-state trajectories directly visualizable.

Two solutions

Analytic sample-wise solution

When the residual-block width is at least the number of training samples, the neuron boundaries can be chosen so that every sample has a distinct activation pattern. The output weights can then assign each sample its own velocity.

This produces an exact zero-loss solution in which every sample follows a prescribed straight trajectory to its target. The construction is highly sample-specific and finely tuned.

Optimized collective solution

Standard endpoint training with Adam finds a qualitatively different solution:

the samples reorganize until the two classes become approximately linearly separable;

the classes then move collectively toward their targets;

only small within-class corrections remain.

Thus, despite having enough capacity to interpolate sample by sample, the trained network discovers shared class-level dynamics.

Evidence for neuron collapse

Near the depth at which the classes become linearly separable, many neurons begin to separate the classes individually. Their activation patterns are also strongly aligned: the activation matrix restricted to these neurons is approximately low rank, with one dominant singular mode.

These neurons therefore behave approximately like copies of one effective class gate. The leading mode generates collective class motion, while subleading modes permit smaller within-class adjustments.

Why redundancy may be preferred

If (M) neurons have identical activation patterns, the network depends only on one collective combination of their output weights. Redistributing this contribution among the neurons leaves the represented function unchanged, producing one collective direction and (M-1) flat relative directions.

Approximate neuron collapse turns these exactly flat directions into weakly curved directions. This predicts that the small Hessian eigenvalues should be controlled by deviations from the shared activation pattern.

Consistent with this picture, the optimized collective solution has:

a more rapidly decaying Hessian spectrum;

fewer resolved quadratic directions at fixed scale;

lower local free energy under isotropic parameter perturbations;

broader low-loss neighborhoods than the analytic sample-wise solution.

The current evidence therefore supports—but does not yet prove—the mechanism

[
\text{neuron redundancy}
;\longrightarrow;
\text{softer loss geometry}
;\longrightarrow;
\text{entropic preference for collective transport}.
]

Main takeaway

Overparameterization does not necessarily bias a network toward sample-wise memorization. In this system, optimization instead finds a redundant representation in which many neurons implement nearly the same class-level operation, creating a broader region of parameter space and transporting samples collectively.

The main open problem is to derive quantitatively how subleading activation modes lift the flat directions and determine the small Hessian eigenvalues.
