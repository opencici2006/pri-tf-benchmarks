#!/bin/bash

REPO_NAMES=("DS2" "SSD" "NMT" "WGAN" "DRAW" "DCGAN" "UNet" "RFCN" 
            "TransformerSpeech" "FasterRCNN" "A3C" "3DUNet" "WaveNet" 
            "MobileNet" "InceptionV4" "WideandDeep" "MaskRCNN" 
            "3DGAN" "DenseNet" "TransformerLanguage" "SqueezeNets" "YoloV2" 
            "InceptionResNetV2")
REPO_PREFIX="git@github.com:NervanaSystems/tensorflow-"
PWD=$(pwd)
TMP_DIR="${PWD}/tmp"
FILE_NAME="platform_util.py"
FILE_PATH="${PWD}/${FILE_NAME}"
BRANCH_NAME="master"
clean_up()
{
  echo "Cleaning up..."
   rm -rf ${TMP_DIR}
}
error_exit()
{
   echo $*
   clean_up
   exit 1
}

echo "Removing existing local repos..."
rm -rf "$TMP_DIR"
mkdir -p "${TMP_DIR}"
for repo in "${REPO_NAMES[@]}"
do
  echo "Updating ${repo}"
  cd "${TMP_DIR}" && mkdir -p "${repo}" && cd "${repo}"
  [ $? -ne 0 ] && error_exit "Error creating directory $(pwd)"
  git clone --depth=1 "${REPO_PREFIX}${repo}.git" .
  [ $? -ne 0 ] && error_exit "Unable to clone repo: ${repo}"
  git checkout ${BRANCH_NAME}
  [ $? -ne 0 ] && error_exit "Unable to checkout ${BRANCH_NAME} branch"
  cp ${FILE_PATH} .
  git add ${FILE_NAME} && git commit -m "Updating ${FILE_NAME}" && git push
  [ $? -ne 0 ] && error_exit "Error pushing to ${repo}"
done

echo "DONE updating ${FILE_NAME}"
clean_up