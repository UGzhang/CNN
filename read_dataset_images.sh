echo "dataset image..."
#!/bin/bash
sh build.sh
exec ./AdvPT $1 $2 $3
