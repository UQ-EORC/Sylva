#!/bin/sh
# Build libsylva_rivlib.so, Sylva's C interface over RiVLib's C++ point-cloud
# class (needed for MTA resolution; see docs/guide/io.md).
#
#   RIVLIB=/path/to/rivlib-2_6_0-x86_64-linux-gcc11 ./build.sh [install-dir]
#
# RiVLib (proprietary, download from RIEGL) is linked statically; the result
# is installed next to libscanifc in $RIVLIB/lib unless an install dir is
# given, and Sylva finds it there or through SYLVA_RIVLIB_SHIM.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
RIVLIB=${RIVLIB:-${RIVLIB_PATH:-${RIVLIB_HOME:-}}}
if [ -z "$RIVLIB" ] || [ ! -d "$RIVLIB/include/riegl" ]; then
  echo "set RIVLIB (or RIVLIB_PATH) to the extracted RiVLib SDK (containing include/riegl and lib/)" >&2
  exit 1
fi
DEST=${1:-$RIVLIB/lib}
CXX=${CXX:-g++}
$CXX -std=c++14 -O2 -fPIC -shared -o "$DEST/libsylva_rivlib.so" "$HERE/sylva_rivlib.cpp" \
  -I"$RIVLIB/include" -L"$RIVLIB/lib" \
  -Wl,--whole-archive -lscanlib-mt -Wl,--no-whole-archive \
  -lriboost_regex-mt -lriboost_filesystem-mt -lriboost_system-mt -lriboost_thread-mt \
  -lriboost_date_time-mt -lriboost_chrono-mt -lriboost_atomic-mt -lpthread
echo "built $DEST/libsylva_rivlib.so"
