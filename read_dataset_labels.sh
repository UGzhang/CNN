echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."
#!/bin/bash
sh build.sh
exec ./AdvPT $1 $2 $3
