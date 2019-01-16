from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import json
import argparse

if __name__ == '__main__':
    
    #1. prepare input parameters from command line.  
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
    parser.add_argument('--AZUREML_SCRIPT_DIRECTORY_NAME', type=str)  

    args = parser.parse_args()

    #2. mount input Azure Blob to dbfs
    mount_dbfs = "/mnt/model/"
    
    #if the mount path already exists, unmount firstly.
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
    print("Azure blob has been mounted to dbfs path", mount_dbfs)

    #3. prepare spark context in databricks.
    spark = SparkSession \
        .builder \
        .getOrCreate()

    #4. load the data to parquet format from mount data path 
    data_path = mount_dbfs + args.data_path
    print("data path mount to dbfs is:", data_path)
    test = spark.read.parquet(data_path)

    #5. load model with PipelineModel from mount model path
    model_path = mount_dbfs + args.model_path
    print("model path mount to dbfs is:", model_path)
    trainedModel = PipelineModel.load(model_path)

    #6. predict test data on model
    prediction = trainedModel.transform(test)
    prediction.head()


