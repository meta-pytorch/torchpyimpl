#!/bin/bash

set -ex

VERSION=$(grep "version" .pyre_configuration | sed -n -e 's/.*\(0\.0\.[0-9]*\).*/\1/p')
if command -v uv >/dev/null 2>&1; then
  uv pip install pyre-check-nightly==$VERSION
else
  pip install pyre-check-nightly==$VERSION
fi
