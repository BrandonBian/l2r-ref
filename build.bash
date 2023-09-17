#!/bin/bash
# ========================================================================= #
# Filename:                                                                 #
#    build.bash                                                             #
#                                                                           #
# Description:                                                              # 
#    Script to build a docker image of a l2r-node                           #
# ========================================================================= #

DIRNAME="l2r/distrib/docker"
TAG="latest";

print_usage()
{
  echo "Builds a docker image of the l2r directory. Usage:"
  echo "  $0 -n <image_name> -d <dockerfile> -t <image_tag>"
}

while [[ $# -gt 0 ]]; do
  opt="$1"
  shift;
  current_arg="$1"
  case "$opt" in
    "-h"|"--help"       ) print_usage; exit 1;;
    "-n"|"--name"       ) NAME="$1"; shift;;
    "-d"|"--dockerfile" ) DOCKERFILE="$1"; shift;;
    "-t"|"--tag"        ) TAG="$1"; shift;;
    *                   ) echo "ERROR: Invalid option: \""$opt"\"" >&2
                          exit 1;;
  esac
done

if [[ "$NAME" == "" || "$DOCKERFILE" == "" ]]; then
  echo "ERROR: Options -n and -d require arguments." >&2
  exit 1
fi

# compress
echo "Compressing l2r directory."
cd ..
tar -czvf l2r.tar.gz l2r
mv l2r.tar.gz $DIRNAME
cp requirements.txt $DIRNAME
cd $DIRNAME

# build image
echo -e "Building image: ${NAME}:${TAG}"
docker build -t ${NAME}:${TAG} -f ${DOCKERFILE} .

# cleanup
rm requirements.txt
rm l2r.tar.gz
echo "Completed build."
