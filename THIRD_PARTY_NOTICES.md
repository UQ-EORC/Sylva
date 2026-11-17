# Third-party notices

Sylva is licensed under the GNU General Public License v3.0 or later (see
`LICENSE`). It contains code adapted from the projects below, whose licences
are compatible with the GPL and require these notices to accompany the source
and any binary distribution (wheels, crates, container images). The methods
Sylva follows without copying code are credited in `docs/references.md`.

## Arbor (r-lidar)

Sylva's terrestrial tree segmentation seeds each tree from the surface of
its trunk rather than from its axis, an idea taken from Arbor
(`src/seed/SeedDetector_cage.cpp`); the code is Sylva's. The tree
segmentation and the quantitative structure models were written with
Arbor's source at hand, and its notice is reproduced here.

Arbor, <https://github.com/r-lidar/arbor>:

```text
Copyright (C) 2026 Jean-Romain Roussel (r-lidar) <info @ r-lidar.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
```

The text of the GNU General Public License is in `LICENSE`.

## lidR

Parts of `sylva.als` are adaptations of the source code of lidR, translated
into Rust and modified (October 2026):

- `crates/sylva-rs/src/als/trees.rs`: `dalponte2016` (lidR
  `src/C_dalponte2016.cpp`), the decision logic of `li2012` and the rules of
  the local maximum filter (lidR `src/LAS.cpp`), with lidR's parameter names
  and defaults (`R/algorithm-its.R`, `R/algorithm-itd.R`).
- `crates/sylva-rs/src/als/metrics.rs`: `entropy`, the cumulative height
  percentages `zpcum`, and the set, names and formulas of the standard
  metrics (lidR `R/metrics_stdmetrics.R`).

The catalogue of tiles processed in buffered chunks follows lidR's design,
not its code. lidR, <https://github.com/r-lidar/lidR>:

```text
Copyright 2016 Jean-Romain Roussel

This file is part of lidR R package.

lidR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
```

lidR's authors and contributors are listed in its `DESCRIPTION`:
Jean-Romain Roussel (author, copyright holder), David Auty, Florian De
Boissieu, Andrew Sánchez Meador, Jean-François Bourdon, Demetrios
Gatziolis, Leon Steinmeier, Stanislaw Adaszewski and Benoît St-Onge.

## CSF (Zhang et al.)

`csf_ground_mask` in `crates/sylva-rs/src/ground.rs` (written in Rust,
September 2026) follows the authors' reference implementation of the cloth
simulation filter in its update scheme and constants (gravity 0.2 x time
step squared, damping 0.99, pinning on contact, the cloth-to-point distance
test). The authors publish that code as the CSF library
(<https://github.com/jianboqi/CSF>, Apache License 2.0) and in CloudCompare's
qCSF plugin, whose notice is:

```text
CLOUDCOMPARE PLUGIN: qCSF

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 or later of the License.

Copyright RAMM laboratory, School of Geography, Beijing Normal University
(http://ramm.bnu.edu.cn/)
Wuming Zhang; Jianbo Qi; Peng Wan; Hongtao Wang
```

## Point Cloud Library (PCL)

`pmf_ground_mask` in `crates/sylva-rs/src/ground.rs` (written in Rust,
September 2026) follows PCL's
`ApproximateProgressiveMorphologicalFilter` in its parameters (`cell_size`,
`max_window`, `slope`, `initial_distance`, `max_distance`) and its sequence
of windows and thresholds:

```text
Software License Agreement (BSD License)

Point Cloud Library (PCL) - www.pointclouds.org
Copyright (c) 2009-2012, Willow Garage, Inc.
Copyright (c) 2012-, Open Perception, Inc.
Copyright (c) 2014, RadiantBlue Technologies, Inc.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of the copyright holder(s) nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## AMAPVox

`crates/sylva-rs/src/voxel/` and `python/sylva/voxels/` follow AMAPVox, built
on raycloudtools functions. What AMAPVox contributes (the path length and
attenuation estimators, the leaf angle distributions and the laser
specifications) is the work of UMR AMAP (botAnique et Modélisation de
l'Architecture des Plantes et des végétations), AMAPVox,
<https://github.com/umr-amap/AMAPVox>. Its repository carries the CeCILL
Free Software License Agreement, version 2.0, as its `LICENSE` file, whose
Article 5.3.4 allows the software to be included in code under the GNU GPL
and the whole distributed under the GPL; its README names the CeCILL-B
licence instead (<https://cecill.info>). Sylva relies on the licence file. Estimators documented in Vincent, G., Pimont, F., & Verley, P. (2021),
*A note on PAD/LAD estimators implemented in AMAPVox 1.7*,
<https://doi.org/10.23708/1AJNMP>.

## raycloudtools (CSIRO)

Sylva follows raycloudtools (<https://github.com/csiro-robotics/raycloudtools>)
in the places listed below, with its source as the reference. raycloudtools
is described in Lowe, T., & Stepanas, K. (2021), RayCloudTools: A concise
interface for analysis and manipulation of ray clouds, *IEEE Access*, *9*,
79712–79724, <https://doi.org/10.1109/ACCESS.2021.3084954>, and its tree
extraction in Devereux et al. (2026), *Remote Sensing of Environment*,
*334*, 115162, <https://doi.org/10.1016/j.rse.2025.115162>.

- `crates/sylva-rs/src/canopy/`: `canopy.density_grid` (its `DensityGrid`
  estimator; Lowe et al. 2021).
- `crates/sylva-rs/src/cluster.rs`: the form of the edge cost of the tree
  segmentation graph, edge length to a power divided by a power of the
  cosine of the bend (`Bend`), and the lateral term and per-tree scaling
  of the shortest-path search (`dijkstra_gravity`, `dijkstra_scaled`);
  raycloudtools `raylib/extraction/raysegment.cpp`.
- `crates/sylva-rs/src/qsm/model.rs`: the power-mean section radius, the
  rule that gives unmeasured segments their radius (in proportion to the
  length they carry, a fork sharing its parent's cross-section) and the
  cap on the children's summed cross-section at a fork
  (`raylib/extraction/raytrees.cpp`).

The raycloudtools licence:

```text
CSIRO Open Source Software Licence Agreement (variation of the BSD / MIT License)
Copyright (c) 2020, Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.
All rights reserved. CSIRO is willing to grant you a licence to this Ray Cloud Tools library on the following terms, except where otherwise indicated for third party material.
Redistribution and use of this software in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of CSIRO nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission of CSIRO.
EXCEPT AS EXPRESSLY STATED IN THIS AGREEMENT AND TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS PROVIDED "AS-IS". CSIRO MAKES NO REPRESENTATIONS, WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY REPRESENTATIONS, WARRANTIES OR CONDITIONS REGARDING THE CONTENTS OR ACCURACY OF THE SOFTWARE, OR OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, THE ABSENCE OF LATENT OR OTHER DEFECTS, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE.
TO THE FULL EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL CSIRO BE LIABLE ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION, IN AN ACTION FOR BREACH OF CONTRACT, NEGLIGENCE OR OTHERWISE) FOR ANY CLAIM, LOSS, DAMAGES OR OTHER LIABILITY HOWSOEVER INCURRED.  WITHOUT LIMITING THE SCOPE OF THE PREVIOUS SENTENCE THE EXCLUSION OF LIABILITY SHALL INCLUDE: LOSS OF PRODUCTION OR OPERATION TIME, LOSS, DAMAGE OR CORRUPTION OF DATA OR RECORDS; OR LOSS OF ANTICIPATED SAVINGS, OPPORTUNITY, REVENUE, PROFIT OR GOODWILL, OR OTHER ECONOMIC LOSS; OR ANY SPECIAL, INCIDENTAL, INDIRECT, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, ACCESS OF THE SOFTWARE OR ANY OTHER DEALINGS WITH THE SOFTWARE, EVEN IF CSIRO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH CLAIM, LOSS, DAMAGES OR OTHER LIABILITY.
APPLICABLE LEGISLATION SUCH AS THE AUSTRALIAN CONSUMER LAW MAY APPLY REPRESENTATIONS, WARRANTIES, OR CONDITIONS, OR IMPOSES OBLIGATIONS OR LIABILITY ON CSIRO THAT CANNOT BE EXCLUDED, RESTRICTED OR MODIFIED TO THE FULL EXTENT SET OUT IN THE EXPRESS TERMS OF THIS CLAUSE ABOVE "CONSUMER GUARANTEES".  TO THE EXTENT THAT SUCH CONSUMER GUARANTEES CONTINUE TO APPLY, THEN TO THE FULL EXTENT PERMITTED BY THE APPLICABLE LEGISLATION, THE LIABILITY OF CSIRO UNDER THE RELEVANT CONSUMER GUARANTEE IS LIMITED (WHERE PERMITTED AT CSIRO'S OPTION) TO ONE OF FOLLOWING REMEDIES OR SUBSTANTIALLY EQUIVALENT REMEDIES:
(a)               THE REPLACEMENT OF THE SOFTWARE, THE SUPPLY OF EQUIVALENT SOFTWARE, OR SUPPLYING RELEVANT SERVICES AGAIN;
(b)               THE REPAIR OF THE SOFTWARE;
(c)               THE PAYMENT OF THE COST OF REPLACING THE SOFTWARE, OF ACQUIRING EQUIVALENT SOFTWARE, HAVING THE RELEVANT SERVICES SUPPLIED AGAIN, OR HAVING THE SOFTWARE REPAIRED.
IN THIS CLAUSE, CSIRO INCLUDES ANY THIRD PARTY AUTHOR OR OWNER OF ANY PART OF THE SOFTWARE OR MATERIAL DISTRIBUTED WITH IT.  CSIRO MAY ENFORCE ANY RIGHTS ON BEHALF OF THE RELEVANT THIRD PARTY.
Third Party Components
The following third party components are distributed with the Software.  You agree to comply with the licence terms for these components as part of accessing the Software.  Other third party software may also be identified in separate files distributed with the Software.
___________________________________________________________________

simple-fft: "beer-ware" permissive licence
___________________________________________________________________

As a condition of this licence, you agree that where you make any adaptations, modifications, further developments, or additional features available to CSIRO or the public in connection with your access to the Software, you do so on the terms of the BSD 3-Clause Licence template, a copy available at: http://opensource.org/licenses/BSD-3-Clause.
```

## SciPy (`scipy.ndimage`)

`feature_transform` and `voronoi_line` in `crates/sylva-rs/src/coreg/ground.rs`
are translated from `scipy/ndimage/src/ni_morphology.c`
(<https://github.com/scipy/scipy>):

```text
Copyright (C) 2003-2005 Peter J. Verveer

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. The name of the author may not be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## GBSeparation (Tian & Li, CC BY 4.0)

The graph-based leaf/wood separation in `crates/sylva-rs/src/qsm/wood.rs`
(`gbs_mask`, `classify_piece`; `leaves.classify_leaf_wood(method="gbs")`) is
translated into Rust from GBSeparation by Zhilin Tian and Shihua Li,
<https://doi.org/10.5281/zenodo.6837613>, licensed under the Creative Commons
Attribution 4.0 International licence
(<https://creativecommons.org/licenses/by/4.0/>). It was modified: networkx
graph operations are replaced by union-find, and growth directions come from
the shortest-path predecessors.

## Rust crates and Python packages

Compiled Sylva binaries link Rust crates under MIT, Apache-2.0, BSD, ISC,
Zlib, CC0 and Unicode licences. `THIRD_PARTY_LICENSES.md` lists every crate
with its copyright notices and licence text, and accompanies the wheels; it
is generated by `scripts/third_party_licenses.py` and is regenerated when
the dependencies change. RIEGL's RiVLib and RDB library are not part of
Sylva and are not distributed with it: they are loaded at run time from a
copy the user provides under RIEGL's own licence.
