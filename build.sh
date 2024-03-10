echo "This script should build your project now..."
#!/bin/bash
if [ ! -d "build" ];then
  mkdir build
  else
  echo rm -f build/*
fi

if [ ! -x "AdvPT" ];then
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make
fi
