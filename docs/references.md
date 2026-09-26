# References

The methods Sylva implements, the datasets it is validated against, and the
software it interoperates with. Cite the original work when you publish
results, not just Sylva. Each entry says where in Sylva it is used.

## Methods

Åkerblom, M., Raumonen, P., Casella, E., Disney, M. I., Danson, F. M., Gaulton,
R., Schofield, L. A., & Kaasalainen, M. (2018). Non-intersecting leaf insertion
algorithm for tree structure models. *Interface Focus*, *8*(2), 20170045.
<https://doi.org/10.1098/rsfs.2017.0045>
— the collision-free leaf insertion that `leaves.add_leaves` does not attempt.

Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray
tracing. In *Eurographics 1987—Technical papers* (pp. 3–10). The Eurographics
Association. <https://doi.org/10.2312/egtp.19871000>
— the voxel walk of `canopy.density_grid` and [`sylva.voxels`](api/voxels.md).

Andrew, A. M. (1979). Another efficient algorithm for convex hulls in two
dimensions. *Information Processing Letters*, *9*(5), 216–219.
<https://doi.org/10.1016/0020-0190(79)90072-3>
— the monotone-chain convex hull behind `trees.convex_hull_area`,
`crown_metrics` and `crown_shape`.

Armston, J., Disney, M., Lewis, P., Scarth, P., Phinn, S., Lucas, R., Bunting,
P., & Goodwin, N. (2013). Direct retrieval of canopy gap probability using
airborne waveform lidar. *Remote Sensing of Environment*, *134*, 24–38.
<https://doi.org/10.1016/j.rse.2013.02.021>
— the 1/n weighting of the echoes of a pulse in `canopy.GapProfile`.

Bailey, B. N., & Mahaffee, W. F. (2017). Rapid, high-resolution measurement of
leaf area and leaf orientation using terrestrial LiDAR scanning data.
*Measurement Science and Technology*, *28*(6), 064006.
<https://doi.org/10.1088/1361-6501/aa5cfd>
— the `bailey` attenuation estimator and triangle-facet *G* in
[`sylva.voxels`](api/voxels.md).

Beaton, A. E., & Tukey, J. W. (1974). The fitting of power series, meaning
polynomials, illustrated on band-spectroscopic data. *Technometrics*, *16*(2),
147–185. <https://doi.org/10.1080/00401706.1974.10489171>
— the Tukey biweight option (`robust="tukey"`) of `sylva.coreg.icp`.

Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D shapes.
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, *14*(2),
239–256. <https://doi.org/10.1109/34.121791>
— iterative closest point, `registration.icp` (`method="point"`) and
`sylva.coreg.icp`.

Calders, K., Armston, J., Newnham, G., Herold, M., & Goodwin, N. (2014).
Implications of sensor configuration and topography on vertical plant profiles
derived from terrestrial LiDAR. *Agricultural and Forest Meteorology*, *194*,
104–117. <https://doi.org/10.1016/j.agrformet.2014.03.022>
— the robust ground plane of `canopy.fit_ground_plane`.

Campbell, G. S. (1990). Derivation of an angle density function for canopies
with ellipsoidal leaf angle distributions. *Agricultural and Forest
Meteorology*, *49*(3), 173–176.
<https://doi.org/10.1016/0168-1923(90)90030-A>
— the ellipsoidal χ and its projection function in
[`sylva.leaves`](api/leaves.md) and the `ellipsoidal` leaf angle
distribution of [`sylva.voxels`](api/voxels.md).

Chen, Y., & Medioni, G. (1992). Object modelling by registration of multiple
range images. *Image and Vision Computing*, *10*(3), 145–155.
<https://doi.org/10.1016/0262-8856(92)90066-C>
— point-to-plane ICP, `registration.icp` (`method="plane"`) and
`sylva.coreg.icp`.

Chetverikov, D., Svirko, D., Stepanov, D., & Krsek, P. (2002). The trimmed
iterative closest point algorithm. In *Proceedings of the 16th International
Conference on Pattern Recognition* (Vol. 3, pp. 545–548). IEEE.
<https://doi.org/10.1109/ICPR.2002.1047997>
— the `trim` of `registration.icp` and the trimmed correspondences of
`sylva.coreg.icp`.

de Wit, C. T. (1965). *Photosynthesis of leaf canopies* (Agricultural Research
Reports No. 663). Pudoc. <https://edepot.wur.nl/187115>
— the planophile, erectophile, plagiophile and extremophile leaf angle
distributions of `sylva.leaves` and `sylva.voxels`.

Dai, X., Liang, X., Qi, H., Chen, J., Wang, X., Wang, X., Zhang, Q., & Zhang,
J. (2024). Automated registration of terrestrial point clouds through ground
overlapping searching in forests. *IEEE Transactions on Geoscience and Remote
Sensing*, *62*, 1–13. <https://doi.org/10.1109/TGRS.2024.3471792>
— registering levelled forest scans on their shared ground: the height of a
`sylva.coreg` stem match is taken from the terrain the two scans share.

Devereux, T., Lowe, T., Rivory, J., Bohn Reckziegel, R., Calders, K., Aryal,
R. R., Eaton, G., Cooper, Z., Levick, S., Phinn, S., & Woodgate, W. (2026).
RayExtract: A fast, scalable method for tree volume reconstruction from
terrestrial laser scanning. *Remote Sensing of Environment*, *334*, 115162.
<https://doi.org/10.1016/j.rse.2025.115162>
— raycloudtools' `rayextract trees`: the least-cost-path segmentation that
`trees.segment_trees` and `trees.merge_branches` follow, and the power-mean
section radius and Leonardo's rule at forks in [`sylva.qsm`](api/qsm.md); also
the raycloudtools baseline of the [tree](benchmarks/trees.md) and
[QSM](benchmarks/qsm.md) benchmarks.

Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm
for model fitting with applications to image analysis and automated
cartography. *Communications of the ACM*, *24*(6), 381–395.
<https://doi.org/10.1145/358669.358692>
— RANSAC circle fits in `trees.detect_stems`, `trees.fit_circle_ransac`,
`trees.dbh_profile`, `trees.detect_buttress`, `quality.stem_noise` and the
stem maps of `sylva.coreg`.

Gibson, S. F. F. (1998). Constrained elastic surface nets: Generating smooth
surfaces from binary segmented data. In *Medical Image Computing and
Computer-Assisted Intervention — MICCAI'98* (pp. 888–898). Springer.
<https://doi.org/10.1007/BFb0056277>
— the surface nets that close `qsm.buttress_mesh`.

Goel, N. S., & Strebel, D. E. (1984). Simple beta distribution representation of
leaf orientation in vegetation canopies. *Agronomy Journal*, *76*(5), 800–802.
<https://doi.org/10.2134/agronj1984.00021962007600050021x>
— the beta fit of the leaf inclination distribution in `sylva.leaves`
and the two-parameter beta distribution of `sylva.voxels`.

Hosoi, F., & Omasa, K. (2006). Voxel-based 3-D modeling of individual trees for
estimating leaf area density using high-resolution portable scanning lidar.
*IEEE Transactions on Geoscience and Remote Sensing*, *44*(12), 3610–3618.
<https://doi.org/10.1109/TGRS.2006.881743>
— the contact-frequency PAD profile `canopy.pad_profile_voxel`.

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of
Mathematical Statistics*, *35*(1), 73–101.
<https://doi.org/10.1214/aoms/1177703732>
— the Huber weights of `quality.stem_noise`, `sylva.coreg.icp`, the
coregistration pose graph and `coreg.refine_joint`.

Jupp, D. L. B., Culvenor, D. S., Lovell, J. L., Newnham, G. J., Strahler, A. H.,
& Woodcock, C. E. (2009). Estimating forest LAI profiles and structural
parameters using a ground-based laser called 'Echidna®'. *Tree Physiology*,
*29*(2), 171–181. <https://doi.org/10.1093/treephys/tpn022>
— the gap-probability profile, and the hinge, linear and weighted PAI
estimators of `canopy.GapProfile`.

Kabsch, W. (1976). A solution for the best rotation to relate two sets of
vectors. *Acta Crystallographica Section A*, *32*(5), 922–923.
<https://doi.org/10.1107/S0567739476001873>
— `registration.kabsch` and the point-to-point step of `sylva.coreg.icp`.

Kåsa, I. (1976). A circle fitting procedure and its error analysis. *IEEE
Transactions on Instrumentation and Measurement*, *IM-25*(1), 8–14.
<https://doi.org/10.1109/TIM.1976.6312298>
— the algebraic circle fit that starts `trees.fit_circle`, the stem detector
and `quality.stem_noise`.

Lang, A. R. G., & Xiang, Y. (1986). Estimation of leaf area index from
transmission of direct sunlight in discontinuous canopies. *Agricultural and
Forest Meteorology*, *37*(3), 229–243.
<https://doi.org/10.1016/0168-1923(86)90033-X>
— the clumping index `GapProfile.clumping`.

Levenberg, K. (1944). A method for the solution of certain non-linear problems
in least squares. *Quarterly of Applied Mathematics*, *2*(2), 164–168.
<https://doi.org/10.1090/qam/10666>
— the pose-graph solver of `sylva.coreg` and the circle and cylinder fits.

Low, K.-L. (2004). *Linear least-squares optimization for point-to-plane ICP
surface registration* (Technical Report TR04-004). Department of Computer
Science, University of North Carolina at Chapel Hill.
<https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf>
— the linearised point-to-plane step of `registration.icp`.

Lowe, T., Moghadam, P., Edwards, E., & Williams, J. (2021). Canopy density
estimation in perennial horticulture crops using 3D spinning lidar SLAM.
*Journal of Field Robotics*, *38*(4), 598–618.
<https://doi.org/10.1002/rob.22006>
— the ray-traced density of `canopy.density_grid` (raycloudtools'
`DensityGrid`) and the formulation behind [`sylva.voxels`](api/voxels.md).

Liu, H., Zhang, X., Xu, Y., & Chen, X. (2020). Efficient coarse registration
of pairwise TLS point clouds using ortho projected feature images. *ISPRS
International Journal of Geo-Information*, *9*(4), 255.
<https://doi.org/10.3390/ijgi9040255>
— the vertical offset of levelled scans from the height difference of their
overlap once aligned horizontally, as `sylva.coreg` sets a stem match's height.

Lu, F., & Milios, E. (1997). Globally consistent range scan alignment for
environment mapping. *Autonomous Robots*, *4*(4), 333–349.
<https://doi.org/10.1023/A:1008854305733>
— the pose graph of `sylva.coreg`.

Marquardt, D. W. (1963). An algorithm for least-squares estimation of
nonlinear parameters. *Journal of the Society for Industrial and Applied
Mathematics*, *11*(2), 431–441. <https://doi.org/10.1137/0111030>
— with Levenberg (1944), the pose-graph solver and the circle and cylinder
fits.

Maurer, C. R., Jr., Qi, R., & Raghavan, V. (2003). A linear time algorithm for
computing exact Euclidean distance transforms of binary images in arbitrary
dimensions. *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
*25*(2), 265–270. <https://doi.org/10.1109/TPAMI.2003.1177156>
— nearest-cell gap filling of the coregistration terrain model, through SciPy's
implementation (see Software).

Miller, J. B. (1967). A formula for average foliage density. *Australian Journal
of Botany*, *15*(1), 141–144. <https://doi.org/10.1071/BT9670141>
— `method="miller"` in `canopy.lai_from_gap_fraction`.

Olofsson, K., Holmgren, J., & Olsson, H. (2014). Tree stem and height
measurements using terrestrial laser scanning and the RANSAC algorithm.
*Remote Sensing*, *6*(5), 4323–4344. <https://doi.org/10.3390/rs6054323>
— an earlier stem detector that fits RANSAC circles in height slices and links
them into stems, the approach `trees.detect_stems` shares.

Pimont, F., Allard, D., Soma, M., & Dupuy, J.-L. (2018). Estimators and
confidence intervals for plant area density at voxel scale with T-LiDAR.
*Remote Sensing of Environment*, *215*, 343–370.
<https://doi.org/10.1016/j.rse.2018.06.024>
— the FPL and PPL attenuation estimators and their bias correction in
`sylva.voxels`.

Pimont, F., Soma, M., & Dupuy, J.-L. (2019). Accounting for wood, foliage
properties, and laser effective footprint in estimations of leaf area density
from multiview-LiDAR data. *Remote Sensing*, *11*(13), 1580.
<https://doi.org/10.3390/rs11131580>
— the beam-section weighted FPL estimator and the finite-leaf-size free path in
`sylva.voxels`.

Raumonen, P., Kaasalainen, M., Åkerblom, M., Kaasalainen, S., Kaartinen, H.,
Vastaranta, M., Holopainen, M., Disney, M., & Lewis, P. (2013). Fast automatic
precision tree models from terrestrial laser scanner data. *Remote Sensing*,
*5*(2), 491–520. <https://doi.org/10.3390/rs5020491>
— TreeQSM, the cylinder-model approach that QSMs such as
[`sylva.qsm`](api/qsm.md) build on.

Rusu, R. B., Marton, Z. C., Blodow, N., Dolha, M., & Beetz, M. (2008). Towards
3D point cloud based object maps for household environments. *Robotics and
Autonomous Systems*, *56*(11), 927–941.
<https://doi.org/10.1016/j.robot.2008.08.005>
— `filters.statistical_outlier_removal`.

Shinozaki, K., Yoda, K., Hozumi, K., & Kira, T. (1964). A quantitative analysis
of plant form—the pipe model theory: I. Basic analyses. *Japanese Journal of
Ecology*, *14*(3), 97–105. <https://doi.org/10.18960/seitai.14.3_97>
— the pipe model that sets unmeasured branch radii in `qsm.build_qsm`.

Smith, D. D., Sperry, J. S., Enquist, B. J., Savage, V. M., McCulloh, K. A., &
Bentley, L. P. (2014). Deviation from symmetrically self-similar branching in
trees predicts altered hydraulics, mechanics, light interception and metabolic
scaling. *New Phytologist*, *201*(1), 217–229.
<https://doi.org/10.1111/nph.12487>
— `path_fraction` in `QSM.metrics`.

Taubin, G. (1995). A signal processing approach to fair surface design. In
*Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive
Techniques (SIGGRAPH '95)* (pp. 351–358). ACM.
<https://doi.org/10.1145/218380.218473>
— smoothing of QSM skeleton nodes and radii, and of buttress meshes.

Tian, Z., & Li, S. (2022). Graph-based leaf–wood separation method for
individual trees using terrestrial lidar point clouds. *IEEE Transactions on
Geoscience and Remote Sensing*, *60*, 1–11.
<https://doi.org/10.1109/TGRS.2022.3218603>
— the default `leaves.classify_leaf_wood` labeller (GBSeparation),
translated from the authors' code (see Software).

Tremblay, J.-F., & Béland, M. (2018). Towards operational marker-free
registration of terrestrial lidar data in forests. *ISPRS Journal of
Photogrammetry and Remote Sensing*, *146*, 430–435.
<https://doi.org/10.1016/j.isprsjprs.2018.10.011>
— vertical error dominates stem-based registration, traced to the terrain
under the stems; why `sylva.coreg` takes height from the shared ground.

Umeyama, S. (1991). Least-squares estimation of transformation parameters
between two point patterns. *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, *13*(4), 376–380. <https://doi.org/10.1109/34.88573>
— the reflection-safe rotation of `registration.kabsch` and `sylva.coreg`.

Verroust, A., & Lazarus, F. (2000). Extracting skeletal curves from 3D
scattered data. *The Visual Computer*, *16*(1), 15–25.
<https://doi.org/10.1007/PL00007210>
— the geodesic level-set skeleton of `qsm.build_qsm`.

Vicari, M. B., Disney, M., Wilkes, P., Burt, A., Calders, K., & Woodgate, W.
(2019). Leaf and wood classification framework for terrestrial LiDAR point
clouds. *Methods in Ecology and Evolution*, *10*(5), 680–694.
<https://doi.org/10.1111/2041-210X.13144>
— the path-frequency cue of `qsm.wood_points`.

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

Wang, X., Yang, Z., Cheng, X., Stoter, J., Xu, W., Wu, Z., & Nan, L. (2023).
GlobalMatch: Registration of forest terrestrial point clouds by global
matching of relative stem positions. *ISPRS Journal of Photogrammetry and
Remote Sensing*, *197*, 71–86. <https://doi.org/10.1016/j.isprsjprs.2023.01.013>
— the 4-degree-of-freedom stem registration of levelled scans, and the
sensitivity of stem heights to the terrain model, behind `sylva.coreg`'s
terrain-based height.

Weinmann, M., Jutzi, B., Hinz, S., & Mallet, C. (2015). Semantic point cloud
interpretation based on optimal neighborhoods, relevant features and efficient
classifiers. *ISPRS Journal of Photogrammetry and Remote Sensing*, *105*,
286–304. <https://doi.org/10.1016/j.isprsjprs.2015.01.016>
— the eigenvalue planarity and linearity of `filters.planarity_linearity`.

Wilkes, P., Disney, M., Armston, J., Bartholomeus, H., Bentley, L., Brede, B.,
Burt, A., Calders, K., Chavana-Bryant, C., Clewley, D., Duncanson, L., Forbes,
B., Krisanski, S., Malhi, Y., Moffat, D., Origo, N., Shenkin, A., & Yang, W.
(2023). TLS2trees: A scalable tree segmentation pipeline for TLS data.
*Methods in Ecology and Evolution*, *14*(12), 3083–3099.
<https://doi.org/10.1111/2041-210X.14233>
— the cluster-gap edge weighting tried and rejected in the
[tree benchmark](benchmarks/trees.md).

Wilson, J. W. (1960). Inclined point quadrats. *New Phytologist*, *59*(1),
1–7. <https://doi.org/10.1111/j.1469-8137.1960.tb06195.x>
— the leaf projection function *G* (`LeafAngleDistribution.g`).

Wilson, J. W. (1963). Estimation of foliage denseness and foliage angle by
inclined point quadrats. *Australian Journal of Botany*, *11*(1), 95–105.
<https://doi.org/10.1071/BT9630095>
— the hinge angle (57.5°) used by `canopy.lai_from_gap_fraction` and
`GapProfile`.

Xu, H., Gossett, N., & Chen, B. (2007). Knowledge and heuristic-based modeling
of laser-scanned trees. *ACM Transactions on Graphics*, *26*(4), 19.
<https://doi.org/10.1145/1289603.1289610>
— the geodesic-shell clustering behind `qsm.build_qsm`.

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

Cherlet, W., Dayal, K., Chen, S., Cooper, Z., Disney, M., Hanzl, A., Levick, S.,
Nightingale, J., Origo, N., Senf, C., Soenens, L., Terryn, L., Van den Broeck,
W. A. J., & Calders, K. (2026). Benchmarking tree instance segmentation of
terrestrial laser scanning point clouds. *ISPRS Journal of Photogrammetry and
Remote Sensing*, *231*, 230–247. <https://doi.org/10.1016/j.isprsjprs.2025.10.033>
— the evaluation protocol and published baselines of the
[tree detection benchmark](benchmarks/trees.md).

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

Wielgosz, M., Puliti, S., & Astrup, R. (2026). *SegmentAnyTreeV2: Scaling
transformer-based tree instance segmentation across sensors, platforms, and
forests* (arXiv:2606.08206) [Preprint]. arXiv.
<https://doi.org/10.48550/arXiv.2606.08206>
— published F1 on the four benchmark plots, quoted in the
[tree benchmark](benchmarks/trees.md).

The canopy benchmark also uses TERN AusCover / Ecosystem Surveillance plot
scans and hemispherical photography, which are not deposited per plot; see
<https://www.tern.org.au> for access and citation.

## Formats

American Society for Photogrammetry and Remote Sensing. (2019). *LAS
specification 1.4 – R15*. <https://github.com/ASPRSorg/LAS>
— `.las` and `.laz` reading and writing, including typed extra bytes, in
[`sylva.io`](api/io.md).

Isenburg, M. (2013). LASzip: Lossless compression of LiDAR data.
*Photogrammetric Engineering & Remote Sensing*, *79*(2), 209–217.
<https://doi.org/10.14358/PERS.79.2.209>
— `.laz` compression, through the `las` and `laz` crates.

## Software

Lowe, T. D., & Stepanas, K. (2021). RayCloudTools: A concise interface for
analysis and manipulation of ray clouds. *IEEE Access*, *9*, 79712–79724.
<https://doi.org/10.1109/ACCESS.2021.3084954>
— ray clouds, which Sylva reads and writes, and `rayextract trees`, which it
follows and is benchmarked against:
<https://github.com/csiro-robotics/raycloudtools>.

AMAPVox (UMR AMAP) — voxelisation of lidar data, the reference implementation
of the attenuation estimators: <https://github.com/umr-amap/AMAPVox>.
[`sylva.voxels`](api/voxels.md) is a port of rayvoxel, J. Rivory's unpublished
reimplementation of AMAPVox on raycloudtools; see `THIRD_PARTY_NOTICES.md`. Its
estimators are documented in Vincent, G., Pimont, F., & Verley, P. (2021). *A
note on PAD/LAD estimators implemented in AMAPVox 1.7*. DataSuds.
<https://doi.org/10.23708/1AJNMP>

pylidar-tls-canopy (J. Armston) — Jupp gap-probability profiles from RIEGL and
LEAF instruments, the reference for the [canopy
benchmark](benchmarks/canopy.md): <https://github.com/armstonj/pylidar-tls-canopy>.

Tian, Z., & Li, S. (2022). *Graph-based leaf–wood separation method for
individual trees using terrestrial lidar point clouds: GBSeparation—a python
package* [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.6837613>
— the reference implementation (CC BY 4.0) that
`leaves.classify_leaf_wood(method="gbs")` translates into Rust.

arbor (J.-R. Roussel, r-lidar) — tree segmentation and QSMs for point clouds;
the idea of seeding the trunk surface rather than its axis in
`trees.segment_trees` (`seed_ring`) came from it:
<https://github.com/r-lidar/arbor>.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T.,
Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der
Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R.
J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy
1.0: Fundamental algorithms for scientific computing in Python. *Nature
Methods*, *17*(3), 261–272. <https://doi.org/10.1038/s41592-019-0686-2>
— `scipy.ndimage`'s feature transform, which the coregistration terrain model
transcribes (BSD-3-Clause; see `THIRD_PARTY_NOTICES.md`).

Segfix — a GUI for correcting instance segmentation of tree point clouds; it
opens the `tree_id` column Sylva writes and saves it back in place:
<https://github.com/tim-devereux/segfix>.
