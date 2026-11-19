# Live initial positions

Live Gazebo now selects a new random 32-bit seed when --seed is omitted.
The resolved seed is printed before launch and saved in manifest.json settings;
the inference worker also reports its effective seed. An explicit --seed 42
retains the old reproducible initial state and network sampling. Use the printed
seed across all profiles for paired communication comparisons.

The frozen delivery task uses independent dispersed sampling, not a prepared
triangle: x and y in [-1.5, 1.5] m, heading in [-pi, pi], initial velocities zero,
and pairwise center separation at least 0.7 m. The synthetic circular radius
is 0.25 m, giving at least 0.2 m initial geometric clearance. This is not a
validated footprint or physical protective margin for the actual robots.

The bounded square already limits initial pairwise distance to at most
sqrt(3^2+3^2), approximately 4.24 m. No extra pairwise maximum was introduced;
there is no measured radio range for the laboratory robots yet. Increasing the
spawn area or selecting new distance limits changes the test distribution and
needs separate evaluation. Bounded random initialization does not guarantee
successful formation for every sample.

Run: bash scripts/run_delivery_gazebo.sh good
Repeat: bash scripts/run_delivery_gazebo.sh good --seed PRINTED_SEED
Policy weights, dynamics, spawn bounds and mission criteria are unchanged.
