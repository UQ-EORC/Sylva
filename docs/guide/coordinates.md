# Coordinates

TLS data changes frame several times on its way to a result: from the
scanner's own frame to the project frame (the SOP), sometimes to a projected
map grid, and often back to a local origin so that coordinates in the
millions of metres keep their precision. `sylva.geo.coords` and three
`PointCloud` methods cover these steps:

| Task | Use |
|---|---|
| Shift, rotate, move to a local origin | `cloud.translate`, `cloud.rotate`, `cloud.recentre` |
| Apply a 4x4 matrix | `cloud.transform` (unchanged) |
| Change map projection or datum | `coords.reproject` |
| Register a whole survey from matrix files | `coords.apply_transforms` |
| The same from the shell | `sylva reproject`, `sylva transform` ([Command line](cli.md)) |

## A worked example

A plot in GDA2020 / MGA zone 55, moved to a local origin for processing,
turned to face north-east, and returned to map coordinates and to latitude
and longitude:

```python
import numpy as np
import sylva
from sylva.geo import coords

rng = np.random.default_rng(0)
xyz = rng.uniform([512_000, 5_412_000, 100], [512_050, 5_412_050, 130], (10_000, 3))
plot = sylva.PointCloud(xyz, crs="EPSG:7855")        # or sylva.read("plot.laz")

local, offset = plot.recentre()          # offset = [512000, 5412000, 100]
turned = local.rotate(45, about=(25, 25, 0))
back = turned.rotate(-45, about=(25, 25, 0)).translate(*offset)
np.abs(back.xyz - plot.xyz).max()        # 0.0 here; at most float rounding

lonlat = coords.reproject(plot, "EPSG:7844")        # GDA2020 longitude, latitude
lonlat.crs                                          # 'EPSG:7844'
coords.transformation("EPSG:7855", "EPSG:7844")     # kind='conversion', exact=True
sylva.write(lonlat, "plot_lonlat.laz")              # the CRS goes into the header
```

## CRS on a point cloud

`PointCloud.crs` holds the coordinate reference system of `xyz`, or None
when it is unknown (the default, so clouds built without one behave as
before). It is a string: an EPSG code (`"EPSG:7855"`, or `"EPSG:7855+5711"`
for MGA zone 55 with AHD heights), a PROJ string or WKT. An integer is
turned into `"EPSG:n"`.

- **Reading.** `sylva.read` sets it for LAS/LAZ files from the header's OGC
  WKT record, or else from its GeoTIFF keys (as `"EPSG:n"`). PLY and text
  files carry no CRS, so set it yourself: `cloud.crs = "EPSG:7855"`.
- **Writing.** `sylva.write` stores it in LAS/LAZ files as an OGC WKT record
  (LAS 1.4), which PDAL, LAStools, CloudCompare and QGIS read. A CRS given
  as a PROJ string has no WKT form here and is left out with a warning.
  Clouds whose CRS is geographic are written with a coordinate scale of
  1e-7 degrees (about 1 cm) instead of 1 mm.
- **Carrying.** Subsets, copies, `with_attrs`, `transform`, `translate`,
  `rotate`, `recentre`, thinning and `concatenate` keep it unchanged.
  `concatenate` refuses clouds in different CRSs.
- **Meaning after a shift.** Because the CRS is kept through `translate`
  and `recentre`, a recentred cloud still names its map CRS: it describes
  the frame the offset returns to. Add the offset back before writing a
  file other software will read as map coordinates, or before reprojecting.

`coords.crs_info` shows what Sylva makes of a definition:

```python
info = coords.crs_info(cloud.crs)
info.label, info.name        # ('EPSG:7855', 'GDA2020 / MGA zone 55')
info.proj4                   # '+proj=utm +zone=55 +south +ellps=GRS80 +units=m +no_defs'
coords.same_crs(cloud.crs, 7855)
```

## Shifting, rotating and recentring

```python
moved = cloud.translate(10.0, -5.0)                  # dz defaults to 0
turned = cloud.rotate(30)                            # about z, through the origin
tilted = cloud.rotate(-2.5, axis="x", about=cloud.xyz.mean(axis=0))
leaned = cloud.rotate(10, axis=(1, 1, 0))            # any axis direction
local, offset = cloud.recentre()                     # and local.translate(*offset) undoes it
```

- **Angles** are degrees, right-handed: positive is counter-clockwise when
  looking down the axis towards the origin, so `rotate(90)` turns +x into +y.
  This is the convention of `registration.rotation_z`.
- **About a point.** `about` puts the axis through that point. Rotating
  projected coordinates about the origin swings the plot around
  (0, 0), thousands of kilometres away; rotate about the plot centre, or
  recentre first.
- **Exactness.** A translation adds the offset once to each coordinate, so
  shifting back returns the original to within one rounding, and exactly for
  whole-metre shifts of millimetre data. Rotations by multiples of 90
  degrees are exact. Repeated round trips do not drift beyond float rounding
  (about 1e-9 m for coordinates of 5e6 m).
- **Recentring.** The default origin is the minimum corner of the cloud
  rounded down to whole metres, so the offset is short and exact. Float32
  holds about seven significant digits: 5412345.678 m is stored to the
  nearest 0.5 m in float32, while the local 45.678 m is kept to 4
  micrometres. Recentre before writing formats or using tools that work in
  float32.
- **Matrices.** `coords.translation_matrix` and `coords.rotation_matrix`
  give the same operations as 4x4 matrices, to compose with registration
  results: `cloud.transform(coords.rotation_matrix(30, about=c) @ T)`.

Attributes are carried over unchanged, so direction-like attributes (`nx`,
`ny`, `nz` of ray clouds) are not rotated.

## Reprojecting

```python
mga = coords.reproject(cloud, "EPSG:7855")                    # source from cloud.crs
utm = coords.reproject(cloud, 32755, src_crs="EPSG:7855")     # or given explicitly
xy = coords.reproject(np.array([[147.5, -42.9]]), "EPSG:7855", "EPSG:7844")   # arrays too
```

Definitions of every horizontal CRS in the EPSG registry (codes up to 65535)
are built in, and the transformation runs in the Rust core with proj4rs, a
Rust port of PROJ.4, so neither PROJ nor GDAL is needed. WKT without an EPSG
code (ESRI `.prj` files, for example) is converted for transverse Mercator,
Lambert conformal conic, Albers, Mercator, Lambert azimuthal equal area,
polar and oblique stereographic and equirectangular projections.
Geographic coordinates are x = longitude and y = latitude, in degrees.

How exact the result is depends on the datums:

| Transformation | Example | Result |
|---|---|---|
| Same datum, other projection | MGA zone 55 to GDA2020 latitude and longitude; UTM to WGS 84 | exact to float rounding; z unchanged |
| Datums tied to WGS 84 by zero parameters | GDA94, NAD83, ETRS89 or NZGD2000 to WGS 84 | treated as the same datum, as PROJ's default does |
| Helmert (`towgs84`) datum change | OSGB36 or DHDN to WGS 84 | exact to the published parameters; z changes |
| Datums without parameters between them | GDA2020 to WGS 84 or to GDA94 | approximate: the datum shift is not applied |
| Grid-based datum change | NAD27 (NADCON), NTv2 grids | refused with a `ValueError` |
| Vertical datum change | AHD to ellipsoidal or AVWS heights | approximate: heights are not changed |

`coords.transformation(src, dst)` reports which case applies before
anything is transformed (`kind`, `exact`, `changes_z` and a `note`).
An approximate reprojection issues a `coords.ApproximateTransformationWarning`
naming what was not applied; turn it into an error with
`warnings.simplefilter("error", coords.ApproximateTransformationWarning)`
where a silent metre matters.

The reference checks in the test suite: the GDA technical manual's Flinders
Peak example (MGA zone 55 to 1 mm), UTM and Albers coordinates computed by
PROJ 9.8, and PROJ's OSGB36 and GDA94 to GDA2020 Helmert results to 1e-9
degrees.

Things to know:

- **Heights.** A Helmert datum change treats z as ellipsoidal height and
  changes it with the datum. TLS heights are usually orthometric (AHD) or
  local; if so, keep the original z (`out.xyz[:, 2] = cloud.z`), since
  converting between height systems needs a geoid model, which is not
  available.
- **GDA94 and GDA2020.** The EPSG definition of GDA2020 carries no
  parameters, so GDA94 to GDA2020 is a null transformation (off by about
  1.5 m). Give GDA2020 as a PROJ string with the published Helmert
  (EPSG:8048) to apply it; this reproduces PROJ's result to 0.1 mm:

    ```python
    gda2020 = ("+proj=utm +zone=55 +south +ellps=GRS80 +units=m +no_defs "
               "+towgs84=-0.06155,0.01087,0.04019,-0.0394924,-0.0327221,-0.0328979,0.009994")
    mga2020 = coords.reproject(cloud_mga94, gda2020, src_crs="EPSG:28355")
    mga2020.xyz[:, 2] = cloud_mga94.z       # keep AHD heights
    mga2020.crs = "EPSG:7855"               # label it with the code once shifted
    ```

- **Round trips through a Helmert change** apply the negated parameters on
  the way back (as PROJ.4 did), which returns to the original within 0.1 mm.
- **Out-of-domain points** (and NaN coordinates) become NaN rather than
  failing the whole cloud.
- **Rasters are not reprojected.** A regular grid does not stay regular
  under a change of projection, so moving a `Raster`'s extent would be
  wrong; reproject the points and rebuild the DTM or CHM, or resample the
  GeoTIFF with GDAL's `gdalwarp`. `coords.reproject` raises `TypeError` for
  a `Raster`.

## Applying registration results

`coords.apply_transforms` puts the scans of a survey into one frame with the
matrices a registration produced, and optionally merges them:

```python
# transforms.json from sylva coreg / SurveyResult.save, matched by scan name
clouds = coords.apply_transforms("scans/", "plot_coreg/transforms.json")

# RiSCAN .DAT matrices (DAT/ScanPos001.DAT, ...), merged into one file
merged = coords.apply_transforms(
    ["ScanPos001_2cm.laz", "ScanPos002_2cm.laz"], "plot.RiSCAN/DAT",
    out="plot_registered.laz", merge=True,          # adds scan_id
)

# The SOPs of a RiSCAN project, or matrices in memory
clouds = coords.apply_transforms(paths, "plot.RiSCAN")
clouds = coords.apply_transforms({"north": north, "south": south}, {"north": T1, "south": T2})
clouds = coords.apply_transforms([a, b], [T1, T2], out="registered/")   # by position
```

Transforms can be a `{name: 4x4}` mapping, a list (by position), a single
matrix for every scan, a `transforms.json`, a single matrix file, a
directory of `.DAT` files or a RiSCAN project directory (the SOPs; the POP
is not applied). The existing readers do the parsing:
`coreg.load_transforms`, `io.read_matrix_file` and `read_riscan_project`.

A scan is matched to a named transform by its mapping key or its path: the
file stem, or any folder above it, that equals the name or starts with it
followed by a separator. `ScanPos001.rxp`, `ScanPos001_2cm.laz` and
`ScanPos001/scans/240101_1200.rxp` all take `ScanPos001`; `ScanPos0011`
does not. A scan without a match, or with two, raises `ValueError`.
