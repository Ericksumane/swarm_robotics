# Triangle distance invariant

The requested initial triangle uses 1m center-to-center distance for each of
the three pairs. Its area is sqrt(3)/4 square meters. Distances (1,1,2)m have
zero area and place the robots on a line: equality in the triangle inequality.
They are not a corrected nondegenerate triangle. Preserve (1,1,1)m for the
ongoing training experiment. A line task would need a new distance matrix,
agent ordering, reward/feature contract and separate training.

The extended experiment is paused/resumed at a model checkpoint when necessary;
no mid-run change of geometry or success thresholds is introduced.
The external references and intended reuse boundaries are in repository_review.md.
