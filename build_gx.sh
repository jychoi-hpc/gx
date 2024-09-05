#!/bin/bash

export GX_DIR=$(dirname $0)

if [ -v $GK_SYSTEM ]; then
  if [ -v $1 ]; then
    echo "You must either set GK_SYSTEM or call this script as ./build_gx.sh <SYSTEMNAME>"
    exit 1
  fi
  export GK_SYSTEM=$1
fi

AUTO_MODULES=on
source module-config

cd $GX_DIR;

make clean;
make -j8;
