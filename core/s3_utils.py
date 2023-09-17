# ========================================================================= #
# Filename:                                                                 #
#    s3_utils.py                                                            #
#                                                                           #
# Description:                                                              #
#    AWS s3 utility functions                                               #
# ========================================================================= #

import logging
import boto3
from botocore.exceptions import ClientError

def bucket_exists(bucket_name):
    """Check if an S3 bucket exists

    :param string bucket_name: bucket to create
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        exists = bucket_name in buckets
    except ClientError as e:
        logging.error(e)
        return False
    return exists

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param string bucket_name: bucket to create
    :param string region: region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(file_data, bucket, object_name):
    """Upload a binary file to an S3 bucket

    :param bytes file_data: File to upload
    :param string bucket: Bucket to upload to
    :param string object_name: S3 object name
    :return: True if file was uploaded, else False
    """
    s3_client = boto3.client('s3')
    try:
        response = s3_client.put_object(Body=file_data, Bucket=bucket,
                                        Key=object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
