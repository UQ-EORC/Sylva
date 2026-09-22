# References

The methods Sylva implements, the datasets it is validated against, and the
software it interoperates with. Cite the original work when you publish
results, not just Sylva. Each entry says where in Sylva it is used.

## Methods

Bailey, B. N., & Mahaffee, W. F. (2017). Rapid, high-resolution measurement of
leaf area and leaf orientation using terrestrial LiDAR scanning data.
*Measurement Science and Technology*, *28*(6), 064006.
<https://doi.org/10.1088/1361-6501/aa5cfd>
— the `bailey` attenuation estimator and triangle-facet *G* in
[`sylva.voxels`](api/voxels.md).

Campbell, G. S. (1990). Derivation of an angle density function for canopies
with ellipsoidal leaf angle distributions. *Agricultural and Forest
Meteorology*, *49*(3), 173–176.
<https://doi.org/10.1016/0168-1923(90)90030-A>
— the ellipsoidal χ and its projection function in
[`sylva.leaves`](api/leaves.md).

Goel, N. S., & Strebel, D. E. (1984). Simple beta distribution representation of
leaf orientation in vegetation canopies. *Agronomy Journal*, *76*(5), 800–802.
<https://doi.org/10.2134/agronj1984.00021962007600050021x>
— the beta fit of the leaf inclination distribution in `sylva.leaves`.

Hosoi, F., & Omasa, K. (2006). Voxel-based 3-D modeling of individual trees for
estimating leaf area density using high-resolution portable scanning lidar.
*IEEE Transactions on Geoscience and Remote Sensing*, *44*(12), 3610–3618.
<https://doi.org/10.1109/TGRS.2006.881743>
— the contact-frequency PAD profile `canopy.pad_profile_voxel`.

Jupp, D. L. B., Culvenor, D. S., Lovell, J. L., Newnham, G. J., Strahler, A. H.,
& Woodcock, C. E. (2009). Estimating forest LAI profiles and structural
parameters using a ground-based laser called 'Echidna®'. *Tree Physiology*,
*29*(2), 171–181. <https://doi.org/10.1093/treephys/tpn022>
— the gap-probability profile, and the hinge, linear and weighted PAI
estimators of `canopy.GapProfile`.

Lang, A. R. G., & Xiang, Y. (1986). Estimation of leaf area index from
transmission of direct sunlight in discontinuous canopies. *Agricultural and
Forest Meteorology*, *37*(3), 229–243.
<https://doi.org/10.1016/0168-1923(86)90033-X>
— the clumping index `GapProfile.clumping`.

Lowe, T., Moghadam, P., Edwards, E., & Williams, J. (2021). Canopy density
estimation in perennial horticulture crops using 3D spinning lidar SLAM.
*Journal of Field Robotics*, *38*(4), 598–618.
<https://doi.org/10.1002/rob.22006>
— the ray-traced density formulation behind [`sylva.voxels`](api/voxels.md).

Miller, J. B. (1967). A formula for average foliage density. *Australian Journal
of Botany*, *15*(1), 141–144. <https://doi.org/10.1071/BT9670141>
— `method="miller"` in `canopy.lai_from_gap_fraction`.

Pimont, F., Allard, D., Soma, M., & Dupuy, J.-L. (2018). Estimators and
confidence intervals for plant area density at voxel scale with T-LiDAR.
*Remote Sensing of Environment*, *215*, 343–370.
<https://doi.org/10.1016/j.rse.2018.06.024>
— the FPL and PPL attenuation estimators and their bias correction in
`sylva.voxels`.

Raumonen, P., Kaasalainen, M., Åkerblom, M., Kaasalainen, S., Kaartinen, H.,
Vastaranta, M., Holopainen, M., Disney, M., & Lewis, P. (2013). Fast automatic
precision tree models from terrestrial laser scanner data. *Remote Sensing*,
*5*(2), 491–520. <https://doi.org/10.3390/rs5020491>
— TreeQSM, the cylinder-model approach [`sylva.qsm`](api/qsm.md) follows and is
benchmarked against.

Tian, Z., & Li, S. (2022). Graph-based leaf–wood separation method for
individual trees using terrestrial lidar point clouds. *IEEE Transactions on
Geoscience and Remote Sensing*, *60*, 1–11.
<https://doi.org/10.1109/TGRS.2022.3218603>
— the default `leaves.classify_leaf_wood` labeller (GBSeparation).

Vicari, M. B., Pisek, J., & Disney, M. (2019). New estimates of leaf angle
distribution from terrestrial LiDAR: Comparison with measured and modelled
estimates from nine broadleaf tree species. *Agricultural and Forest
Meteorology*, *264*, 322–333.
<https://doi.org/10.1016/j.agrformet.2018.10.021>
— leaf angle distributions from point normals, in `sylva.leaves` and the
`inclination=True` path of `sylva.voxels`.

Vincent, G., Antin, C., Laurans, M., Heurtebize, J., Durrieu, S., Lavalley, C.,
& Dauzat, J. (2017). Mapping plant area index of tropical evergreen forest by
airborne laser scanning. A cross-validation study using LAI2200 optical sensor.
*Remote Sensing of Environment*, *198*, 254–266.
<https://doi.org/10.1016/j.rse.2017.05.034>
— AMAPVox, whose voxel traversal and outputs `sylva.voxels` mirrors.

Wilson, J. W. (1963). Estimation of foliage denseness and foliage angle by
inclined point quadrats. *Australian Journal of Botany*, *11*(1), 95–105.
<https://doi.org/10.1071/BT9630095>
— the hinge angle (57.5°) used by `canopy.lai_from_gap_fraction` and
`GapProfile`.

Zhang, K., Chen, S.-C., Whitman, D., Shyu, M.-L., Yan, J., & Zhang, C. (2003). A
progressive morphological filter for removing nonground measurements from
airborne LIDAR data. *IEEE Transactions on Geoscience and Remote Sensing*,
*41*(4), 872–882. <https://doi.org/10.1109/TGRS.2003.810682>
— `ground.classify_ground_pmf`.

Zhang, W., Qi, J., Wan, P., Wang, H., Xie, D., Wang, X., & Yan, G. (2016). An
easy-to-use airborne LiDAR data filtering method based on cloth simulation.
*Remote Sensing*, *8*(6), 501. <https://doi.org/10.3390/rs8060501>
— `ground.classify_ground_csf`.

## Validation data

Burt, A., Boni Vicari, M., da Costa, A. C. L., Coughlin, I., Meir, P., Rowland,
L., & Disney, M. (2021). New insights into large tropical tree mass and
structure from direct harvest and terrestrial lidar. *Royal Society Open
Science*, *8*(2), 201458. <https://doi.org/10.1098/rsos.201458>
— felled tropical trees in the [QSM benchmark](benchmarks/qsm.md).

Calders, K., Newnham, G., Burt, A., Murphy, S., Raumonen, P., Herold, M.,
Culvenor, D., Avitabile, V., Disney, M., Armston, J., & Kaasalainen, M. (2015).
Nondestructive estimates of above-ground biomass using terrestrial laser
scanning. *Methods in Ecology and Evolution*, *6*(2), 198–208.
<https://doi.org/10.1111/2041-210X.12301>
— destructively harvested trees in the QSM benchmark.

Calders, K., Verbeeck, H., Burt, A., Origo, N., Nightingale, J., Malhi, Y.,
Wilkes, P., Raumonen, P., Bunce, R., & Disney, M. (2022). *Terrestrial
laser scanning data Wytham Woods: Individual trees and quantitative structure
models (QSMs)* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.7307956>

Cherlet, W., Dayal, K., Chen, S., Cooper, Z., Disney, M., Hanzl, A., Levick, S.,
Nightingale, J., Origo, N., Senf, C., Soenens, L., Terryn, L., Van den Broeck,
W. A. J., & Calders, K. (2025). *TLS forest instance segmentation benchmark:
2983 manually segmented trees from four plots* [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.16875688>
— the four plots of the [tree detection benchmark](benchmarks/trees.md).

Demol, M., Gielen, B., & Verbeeck, H. (2021). *QSMs, point cloud and harvest
data from a destructive forest biomass experiment in Belgium using terrestrial
laser scanning* [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.4557401>

Gonzalez de Tanago, J., Lau, A., Bartholomeus, H., Herold, M., Avitabile, V.,
Raumonen, P., Martius, C., Goodman, R. C., Disney, M., Manuri, S., Burt, A., &
Calders, K. (2018). Estimation of above-ground biomass of large tropical trees
with terrestrial LiDAR. *Methods in Ecology and Evolution*, *9*(2), 223–234.
<https://doi.org/10.1111/2041-210X.12904>
— felled tropical trees in the QSM benchmark.

Hackenberg, J. (2021). *Hackenberg et al 2021* [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.5131717>
— SimpleForest clouds, QSMs and reference table.

Momo Takoudjou, S., Ploton, P., Sonké, B., Hackenberg, J., Griffon, S., de
Coligny, F., Kamdem, N. G., Libalah, M., Mofack, G. I., Le Moguédec, G.,
Pélissier, R., & Barbier, N. (2018). Using terrestrial laser scanning data to
estimate large tropical trees biomass and calibrate allometric models: A
comparison with traditional destructive approach. *Methods in Ecology and
Evolution*, *9*(4), 905–916. <https://doi.org/10.1111/2041-210X.12933>
— felled tropical trees in the QSM benchmark.

Owen, H. J. F., Grieve, S., & Lines, E. R. (2024). *Plot-level semantically
labelled terrestrial laser scanning point clouds* [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.13268500>
— plot-scale leaf/wood labels (CC BY-NC 4.0).

Van den Broeck, W. A. J., Terryn, L., Chen, S., Cherlet, W., Cooper, Z. T., &
Calders, K. (2025). Pointwise deep learning for leaf-wood segmentation of
tropical tree point clouds from terrestrial laser scanning. *ISPRS Journal of
Photogrammetry and Remote Sensing*, *227*, 366–382.
<https://doi.org/10.1016/j.isprsjprs.2025.06.023>
— the manually labelled trees of the leaf/wood benchmark; data set at
<https://doi.org/10.5281/zenodo.13759407>.

The canopy benchmark also uses TERN AusCover / Ecosystem Surveillance plot
scans and hemispherical photography, which are not deposited per plot; see
<https://www.tern.org.au> for access and citation.

## Software

Lowe, T. D., & Stepanas, K. (2021). RayCloudTools: A concise interface for
analysis and manipulation of ray clouds. *IEEE Access*, *9*, 79712–79724.
<https://doi.org/10.1109/ACCESS.2021.3084954>
— ray clouds, `rayextract trees` and `rayvoxel`, which Sylva reads, ports and
is benchmarked against: <https://github.com/csiro-robotics/raycloudtools>.

AMAPVox (UMR AMAP) — voxelisation of lidar data, the reference implementation
of the attenuation estimators: <https://github.com/umr-amap/AMAPVox>.

pylidar-tls-canopy (J. Armston) — Jupp gap-probability profiles from RIEGL and
LEAF instruments, the reference for the [canopy
benchmark](benchmarks/canopy.md): <https://github.com/armstonj/pylidar-tls-canopy>.

GBSeparation — the reference implementation of Tian & Li (2022):
<https://github.com/qforestlab/leaf-wood-segmentation-with-GBSeparation>.

Segfix — a GUI for correcting instance segmentation of tree point clouds; it
opens the `tree_id` column Sylva writes and saves it back in place:
<https://github.com/tim-devereux/segfix>.
