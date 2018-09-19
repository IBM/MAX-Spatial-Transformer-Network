#!/bin/bash
MODEL_ZIP_FILE="spatial_transformer.zip"
DATA_FILE="mnist_sequence1_sample_5distortions5x5.npz"

if [ "$1" == "clean" ]; then
    rm -f $MODEL_ZIP_FILE training-runs.yml training-runs.yml.bak
    exit 0
fi

make_bucket() {
  if [[ $(aws --endpoint-url=http://s3-api.us-geo.objectstorage.softlayer.net s3 mb s3://$1) ]]; then
    return 0
  else
    echo 'Bucket name already exists.'
    return 1
  fi
}

# Check for AWS keys
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
  echo "Please set local environment variables AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY"
  exit 1
fi

# Check for Watson ML credentials
if [ -z "$ML_ENV" ] || [ -z "$ML_USERNAME" ] || [ -z "$ML_PASSWORD" ] || [ -z "$ML_INSTANCE" ]; then
  echo "Please set local environment variables ML_ENV, ML_USERNAME, ML_PASSWORD, and ML_INSTANCE"
  echo "See https://dataplatform.cloud.ibm.com/docs/content/analyze-data/ml_dlaas_environment.html for additional information."
  exit 1
fi

# Zip relevant code to deploy to service
if [ ! -f $MODEL_ZIP_FILE ]; then
  zip $MODEL_ZIP_FILE *.py
fi

# Check if training data is downloaded
if [ ! -f $DATA_FILE ]; then
  echo "Cannot find $DATA_FILE to use for training. Downloading from public S3 bucket."
  curl -O http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/mnist_sequence1_sample_5distortions5x5.npz
fi

# Create training YAML file from template
if [ ! -f ./training-runs.yml ]; then

  CREATE_SUCCESS=1
  while [[ $CREATE_SUCCESS ]]; do
    echo 'Enter a training bucket name'
    read TRAINING_BUCKET
    TRAINING_BUCKET=$(echo $TRAINING_BUCKET|tr -d '\n')
    CREATE_SUCCESS=$(make_bucket $TRAINING_BUCKET)
  done

  echo 'Uploading training data to bucket'
  aws --endpoint-url=http://s3-api.us-geo.objectstorage.softlayer.net s3 cp $DATA_FILE s3://$TRAINING_BUCKET/$DATA_FILE

  CREATE_SUCCESS=1
  while [[ $CREATE_SUCCESS ]]; do
    echo 'Enter a results bucket name'
    read RESULTS_BUCKET
    RESULTS_BUCKET=$(echo $RESULTS_BUCKET|tr -d '\n')
    CREATE_SUCCESS=$(make_bucket $RESULTS_BUCKET)
  donegit s

  echo 'Generating training-runs.yml'
  cp ./training-runs.yml.template ./training-runs.yml
  sed -i .bak "s/    access_key_id:.*/    access_key_id: $AWS_ACCESS_KEY_ID/g" training-runs.yml
  sed -i .bak "s/    secret_access_key:.*/    secret_access_key: $AWS_SECRET_ACCESS_KEY/g" training-runs.yml
  sed -i .bak "s#TRAINING_BUCKET#$TRAINING_BUCKET#" training-runs.yml
  sed -i .bak "s#RESULTS_BUCKET#$RESULTS_BUCKET#" training-runs.yml

fi

ibmcloud ml train $MODEL_ZIP_FILE training-runs.yml
