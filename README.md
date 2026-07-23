# Temporal Tracking of Magnetic Aggregates Under an Alternating Field

Python tools to reconstruct the angular trajectory of particles (magnetic
microparticle aggregates) from microscopy video, and fit their angular
response to an alternating magnetic field.

Developed during my PhD work to study how aggregates of magnetic
microparticles suspended in an aqueous medium respond angularly to an
oscillating magnetic field.

## The Problem

The experiment consists of recording, under a microscope, magnetic particle
aggregates suspended in an aqueous medium while an alternating magnetic field
is applied. Each aggregate rotates/oscillates in response to the field, and
the goal is to measure **how that angle changes over time** for each
particle, and compare that response to the phase of the applied field.

The challenge isn't just measuring an angle: it's reconstructing the
**temporal identity** of each particle across hundreds of frames, in a field
of view where dozens of aggregates are moving simultaneously, appearing,
disappearing, or drifting out of focus.

## Full Pipeline

```
Video  →  Individual frames  →  ImageJ (Analyze Particles)  →  CSV per frame
                                                                     │
                                                                     ▼
                                                   amp_tools_2_07.py
                                                                     │
                      ┌────────────────────────────────────────────────┤
                      ▼                                                ▼
            Tracking + trajectory                          Physical model fit
            reconstruction per particle                    (angular response vs.
                                                              field phase)
```

1. **Acquisition**: the video is split into individual frames.
2. **Per-frame detection (ImageJ)**: each frame is analyzed with ImageJ's
   *Analyze Particles* tool, which fits an ellipse to each detected particle
   and exports a CSV with area, centroid (x, y), ellipse semi-axes (rM, rm),
   and the tilt angle of the major axis.
3. **Temporal tracking (this script)**: for each particle in the reference
   frame, a match is searched for in the next frame by comparing area and
   centroid position within tolerance thresholds. If there is a single
   match, it's considered the same particle, and the process repeats
   frame by frame, reconstructing its full angular trajectory.
4. **Filtering of spurious curves**: reconstructed trajectories are filtered
   by minimum length (particles lost too quickly) and by peak-to-peak angular
   amplitude (particles that don't move, or that jump inconsistently).
5. **Physical fit**: each surviving angular trajectory is fit (via `lmfit`)
   to a model of the response to the alternating field, extracting response
   amplitude, phase relative to the field, and other parameters — both per
   particle and averaged across the whole population.

## How the Tracking Works (the core part)

The core of the algorithm is the `Aggregate` class, representing a particle
detected in a single frame (position, area, angle). Two instances are
considered "the same particle across different frames" if:

- the distance between their centroids is below a threshold `uc` (in pixels), **and**
- the relative area difference is below a threshold `ua` (fraction of area).

When, for a given particle, there is **exactly one** match in the next frame,
it gets linked and the search continues into the following frame. If there's
ambiguity (more than one possible match) or none at all, that particle is
dropped from that trajectory at that point — prioritizing clean trajectories
over completeness.

Each reconstructed trajectory is stored as an `AggregateTime` object, which
accumulates the list of frames, angles, areas, and ellipse radii over time
for that particle.

## Available Fitting Models

The script includes three model functions for the angular response to the
field, selectable depending on the expected behavior of the aggregate:

| Model      | Functional Form                          | Typical Use                                    |
|------------|-------------------------------------------|--------------------------------------------------|
| `seno`     | simple sinusoidal                         | linear response, weak field                       |
| `cuadrada` | square wave                               | saturation/switching-type response                |
| `tansen`   | `arctan(A·sin(...))`                      | nonlinear saturated response (most general case)  |

## Requirements

```
numpy
scipy
matplotlib
lmfit
tqdm
```

## Expected Input Format

One `.csv` file per frame (exported from ImageJ's *Analyze Particles*), with columns:

```
[ _, area, x, y, rM, rm, ..., angle ]
```

(the angle column is always last; the script assumes these positions
based on ImageJ's export convention).

## Basic Usage

```python
# 1. Load each frame already processed by ImageJ
todos = [frame_scan(f, ua, uc) for f in fname]

# 2. Reconstruct temporal trajectories
todos_t = main_loop(todos, frec=frec_redo, f0=f0)

# 3. Define the physical model and initial parameters
params, gmodel = set_params(fase_campo, frec, frate, model='tansen')

# 4. Fit and plot all trajectories found
ang_plot(gmodel, params, todos_t, len(todos), frec, frec_redo, f0, fase_campo)

# 5. (optional) Manually inspect/discard curves
ag_popper(todos_t, ag=[3, 17, 42], pop=True)
```

## Project Status / Known Limitations

This code was written iteratively during my PhD and **is not fully optimized
or refactored**. Known limitations include:

- Variable names mix Spanish and English (inherited from the original lab notes).
- Tracking is "greedy" and frame-by-frame: it does not re-identify particles
  that are temporarily lost (e.g., due to a momentary occlusion).
- Thresholds (`ua`, `uc`, `len_pt`, `dif_max`, `dif_min`) were tuned
  empirically for the specific dataset of this experiment, and likely need
  recalibration for other videos/conditions.
- Some plotting functions (`ang_plot`) have noticeable repetition that could
  be simplified.

Despite this, the pipeline was validated and used to produce results for my
PhD work, successfully reconstructing the angular response of dozens of
aggregates per experiment against the applied magnetic field.

## Author

Nicolás Mele — developed as part of PhD work on the response of magnetic
aggregates to alternating fields. 
