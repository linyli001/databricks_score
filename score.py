from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import json
import argparse

if __name__ == '__main__':

    #input parameters for script to run
    parser = argparse.ArgumentParser()
    parser.add_argument('--asb_account', type=str, dest='asb_account', help='ASB account for access')
    parser.add_argument('--asb_key_name', type=str, dest='asb_key_name', help='ASB key name')
    parser.add_argument('--asb_key', type=str, dest='asb_key', help='ASB private key')
    parser.add_argument('--data_path', type=str, dest='data_path', help='data path on ASB')
    parser.add_argument('--model_path', type=str, dest='model_path', help='model path on ASB')

    #necessary for DatabricksStep Pipeline parameter
    parser.add_argument('--AZUREML_RUN_TOKEN', type=str)
    parser.add_argument('--AZUREML_RUN_ID', type=str)
    parser.add_argument('--AZUREML_ARM_SUBSCRIPTION', type=str)
    parser.add_argument('--AZUREML_ARM_RESOURCEGROUP', type=str)
    parser.add_argument('--AZUREML_ARM_WORKSPACE_NAME', type=str)
    parser.add_argument('--AZUREML_ARM_PROJECT_NAME', type=str)

    args = parser.parse_args()

    mount_dbfs = "/mnt/model/"

    """
    asb_account = "wasbs://azureml@amladbjostoragemtevfszo.blob.core.windows.net/"
    asb_key_name = "fs.azure.account.key.amladbjostoragemtevfszo.blob.core.windows.net"
    asb_key = "ofSDgi91/2+/TP3oDp32UNAodi/wYyAvE7Z2iPvkCkDa39EZUbxbB8ceGNk9DMQ58SYCQfrboMvGNiTTaGkNBg=="
    data_path = '%s/data' %(mount_dbfs)
    model_path = "%s/ExperimentRun/103af909-6d8b-4b48-a219-7a0e15e4b293/outputs/" % (mount_dbfs)
    """

    # mount input Azure Blob to dbfs
    try:
        dbutils.fs.ls(mount_dbfs)
    except Exception:
        print("mnt/model not exist")
    else:
        dbutils.fs.unmount(mount_dbfs)

    dbutils.fs.mount(
      source = args.asb_account,
      mount_point = mount_dbfs,
      extra_configs = {args.asb_key_name: args.asb_key}
    )
    print("asb has been mounted to dbfs")

    #2. load test data
    spark = SparkSession \
        .builder \
        .getOrCreate()

    data_path = mount_dbfs + args.data_path
    print("data path mount to dbfs is:", data_path)
    test = spark.read.parquet(data_path)

    #3. load test model
    model_path = mount_dbfs + args.model_path
    print("model path mount to dbfs is:", model_path)
    trainedModel = PipelineModel.load(model_path)

    #4. predict test data on model
    prediction = trainedModel.transform(test)
    prediction.head()


