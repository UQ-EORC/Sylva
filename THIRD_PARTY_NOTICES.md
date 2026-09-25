# Third-party notices

Sylva is licensed under the GNU General Public License v3.0 or later (see
`LICENSE`). It contains code adapted from the projects below, whose licences
are compatible with the GPL and require these notices to accompany the source
and any binary distribution (wheels, crates, container images). The methods
Sylva follows without copying code are credited in `docs/references.md`.

## AMAPVox, through rayvoxel

`crates/sylva-rs/src/voxel/` and `python/sylva/voxels.py` are a port of
rayvoxel, an unpublished tool by Josh Rivory that reimplements AMAPVox on
raycloudtools functions. What AMAPVox contributes (the path length and
attenuation estimators, the leaf angle distributions and the laser
specifications) is the work of UMR AMAP (botAnique et Modélisation de
l'Architecture des Plantes et des végétations), AMAPVox,
<https://github.com/umr-amap/AMAPVox>, distributed under the CeCILL-B licence
(<https://cecill.info/licences/Licence_CeCILL-B_V1-en.html>) according to its
README. Estimators documented in Vincent, G., Pimont, F., & Verley, P. (2021),
*A note on PAD/LAD estimators implemented in AMAPVox 1.7*,
<https://doi.org/10.23708/1AJNMP>.

## raycloudtools (CSIRO)

Sylva follows raycloudtools (<https://github.com/csiro-robotics/raycloudtools>)
in `canopy.density_grid` (its `DensityGrid` estimator) and in the power-mean
section radius of `qsm.build_qsm` (`raytrees.cpp`). The raycloudtools licence:

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

`feature_transform` and `voronoi_line` in `crates/sylva-rs/src/coreg_ground.rs`
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
Zlib, CC0 and Unicode licences; `cargo metadata` lists each crate's licence,
and their texts are in the crates' sources. RIEGL's RiVLib is not part of
Sylva: it is loaded at run time from a copy the user provides.
