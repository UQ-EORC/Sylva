# Change detection

`sylva.change` compares two epochs of a plot. The principle throughout is
to separate real change from noise and from not having seen something:
every change carries an uncertainty or a level of detection, and anything
the data cannot support (space that one epoch did not observe, model parts
filled in by priors) is labelled rather than reported as change.

## Aligning epochs

## Matching trees

## Tree increments and plot summary

## Point and voxel change

## QSM change

`compare_qsms` compares two cylinder models of one tree, the earlier `a`
and the later `b`, in one frame (align the epochs first). It reports the
stem radius change by height, branch matching with lost, new and grown
branches, volume change by branch order, and height, DBH and crown change.

```python
from sylva import change, qsm, voxels

model_a = qsm.build_qsm(wood_2020, base_xy=(t.x, t.y))
model_b = qsm.build_qsm(wood_2025, base_xy=(t.x, t.y))       # already aligned to 2020
grid_b = voxels.ray_voxelize(shots_2025, 0.1, occlusion=True)  # every 2025 pulse, misses included

c = change.compare_qsms(model_a, model_b, grid_b=grid_b)
c.taper_increment, c.taper_sigma          # mean stem radius increment (m) and its uncertainty
c.taper["increment"], c.taper["trusted"]  # the increment profile, one value per 1 m of height
c.trusted_change, c.untrusted_change      # volume change (m3) both models measured, and the rest
c.orders                                  # the same split for the stem and each branch order
c.branches("lost")["volume"].sum()        # branches gone, their space seen empty in 2025
c.branches("unobserved")                  # missing from the 2025 model, but 2025 did not see there
c.summary()
```

**Trust.** A cylinder is measured when its radius was fitted to points
(`n_points > 0`); the rest of a model comes from the taper and pipe-model
priors of [`build_qsm`](qsm.md) and says little about change. A stem bin is
trusted when both models fitted at least `min_fits` cylinders and
`min_measured` of its length, and when its increment is in line with the
other bins (within `clip` of its own uncertainty from the mean); a branch
pair is trusted when both models measured `min_measured` of its length. The
volume change of every trusted part is `trusted_change`, the rest
`untrusted_change`; they add up to the total, per branch order as well.

**Stem.** In each height bin a line through each model's measured stem
radii gives the radius at the bin centre; their difference is the radius
increment. Its uncertainty is the scatter about both lines plus
`radius_sigma` (1 mm per fit by default), not divided by the number of
cylinders, since the QSM smooths neighbouring radii together. The taper
increment is the weighted mean over the trusted bins, its uncertainty
scaled up by the Birge ratio when the bins scatter more than their
uncertainties allow. A fitted bin far out of line is almost always a
radius the QSM regularised, typically the stem tip tapered to its apex;
it is reported, left out of the mean and counted as untrusted volume.

**Branches.** Branches are matched one order at a time, parents first. A
pair is admissible when the bases lie within `max_base_distance` (0.5 m)
and the directions over the first metre within `max_angle` (35 degrees);
cheaper pairs (distance and angle, each over its limit, plus
`parent_penalty` when the parents are not matched to each other) are taken
first. A matched pair reports its length, volume and radius change and the
shift of its tip. An unmatched branch of the earlier model is lost, one of
the later model new.

**Lost or unobserved.** A branch missing from the later model has either
gone or was not seen. With a ray-traced grid of the later epoch
([`ray_voxelize`](voxels.md), `occlusion=True`) each lost branch is
sampled every half voxel along its axis, leaving out the part inside its
parent: when fewer than `min_observed` of the samples lie in voxels the
later pulses crossed or ended in, the branch is `"unobserved"`; when more
than `max_filled` of the observed samples hold returns, the branch is
`"present"` (still there, but not in the later model); otherwise it is
`"lost"`, and only then does its volume count as trusted loss. A grid of
the earlier epoch (`grid_a`) checks new branches the same way. An
unobserved branch also leaves the crown change untrusted, since the crown
is outlined by the branches.

For a plot, `compare_plot_qsms` pairs the models of matched trees and
tabulates the changes. The tree match is plain data: survivor pairs of tree
ids, and lists of dead and recruited trees.

```python
plot = change.compare_plot_qsms(
    qsms_2020, qsms_2025,                  # PlotQSMs or {tree_id: QSM}
    pairs=[(1, 4), (2, 5), (3, 7)],        # survivors: (id in 2020, id in 2025)
    deaths=[6], recruits=[9],
    grid_b=grid_2025,
)
plot.totals                                # growth (and its trusted part), mortality, recruitment, net
plot.to_csv("qsm_change.csv")              # one row per tree
plot.changes[(1, 4)].taper                 # the full comparison of a survivor
```

A dead tree counts its whole earlier volume as mortality and a recruit its
whole later volume as recruitment, the measured part as trusted.

**Validation.** A synthetic tree (a 10 m tapered stem and five limbs as
point surfaces) was scanned from three positions in each epoch with
`synthetic.scan` (single returns, so foliage is opaque). Between the
epochs the stem thickened by 10 mm along its length, one limb was cut, one
grew 1 m longer, and one was wrapped in a sleeve of foliage that hid it
from every later scan. Over five noise draws:

| Quantity | Truth | Recovered |
|---|---|---|
| Taper increment | 10 mm | 10.1 to 10.3 mm, stated uncertainty 0.6 mm |
| DBH change | 20 mm | 20.1 to 20.8 mm |
| Cut limb | lost | lost in every draw; volume +8 % (the QSM's own bias) |
| Extended limb | +1.00 m | matched, +0.95 to +1.00 m |
| Hidden limb | unobserved | unobserved in every draw; its volume untrusted |
| Stem volume change over the trusted bins | 0.065 to 0.069 m³ | within 7 % |
| Trusted volume change | 0.056 to 0.060 m³ | within 0.8 of the stated uncertainty (5 to 8 L) |

The main error in the trusted total is a limb whose weak fits the QSM
replaced by its allometric prior in one epoch but not the other: its
volume changed by 5 to 7 L although the limb did not change, and the
cylinders still count as measured because the QSM keeps their point count
when it replaces the radius. Treat trusted growth of thin branches that
exceeds its `volume_sigma` by far with care.

Limitations: both models must be in one frame, and the matching is greedy
rather than a global assignment, so a branch that forked differently in
the two models can show as one lost and two new branches. Heights, lengths
and crowns come from the skeleton, which always follows points; the
measured label concerns radii, and so volumes and the taper increment.
Height is trusted when a measured cylinder reaches within `top_band` of the
top in both models, the crown when both models measured most of their
branch length and no unmatched branch went unobserved.
