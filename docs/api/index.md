# Function index

Every public class, method and function, grouped by module. Each name links
to its full entry: parameters with units and defaults, return values,
exceptions, and notes on accuracy where the method has been benchmarked.

Conventions used throughout:

- Lengths are metres, areas m², volumes m³, densities m² m⁻³ (area) or
  m³ m⁻³ (wood volume). Angles are degrees unless a parameter says radians.
- Point arrays are `(N, 3)` float64. Voxel arrays are `(nz, ny, nx)`, and
  grid `shape` properties are `(nx, ny, nz)`.
- Functions return new objects and never modify their inputs. The exceptions
  are `GapProfile.add_scan`, `RayVoxelGrid.add_wood_volume` and
  `trees.tree_heights`, which say so.
- Anything random takes a `seed`, so a run can be repeated exactly.
- Errors: bad files raise `OSError`, bad arguments raise `ValueError`, and
  missing attributes raise `KeyError` or `ValueError` with the attribute's
  name.

<!-- function-index -->
