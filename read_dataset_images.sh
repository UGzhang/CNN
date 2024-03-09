echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."
#!/bin/bash
cd extract-data
mkdir build
cd build
cmake ..
make
cd ../../
exec ./MnistData $1 $2 $3
rm -rf build