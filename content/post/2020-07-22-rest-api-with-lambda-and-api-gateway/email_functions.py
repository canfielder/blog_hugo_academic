'''
This python script documents the functions which will be used to process 
and generate features based on the email dataset.
'''

########## LIBRARIES #############################
import pyspark
import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import HiveContext
#hive_context = HiveContext(spark)

import sys
import os
import datetime
import time

import pandas as pd
import numpy as np

from source.misc_functions import policy_number

########## INITIALIZE SPARK ######################
spark = pyspark.sql.SparkSession.builder.enableHiveSupport().getOrCreate()

########## FUNCTIONS #############################
# Processing
def create_email_id(spark_df):
    '''
    The purpose of this function is to create a unique ID for each email. The function 
    takes a spark DataFrame which requires the following columns:
    
    - VENDOR_REC_RECVD_DT
    - VENDOR_LAUNCH_ID
    - VENDOR_CD
    
    VENDOR_REC_RECVD_DT is converted to string, and then the listed columns
    are concatenated into the new column labeled EMAIL_ID. 
    
    ACCEPTS:
    One dataframe: 
     - Email Data
     
     RETURNS:
     One dataframe:
     - Email data with EMAIL_ID column added
    '''
    
    spark_df_output = spark_df \
        .withColumn('EMAIL_ID', F.concat(F.col('VENDOR_REC_RECVD_DT').cast('string'),
                                         F.col('VENDOR_LAUNCH_ID'),
                                         F.col('VENDOR_CD')
                                        )
                   )
    
    return spark_df_output


def campaign_event_filter(df):
    '''
    The purpose of this function to filter out any campaigns from the Email dataset where the following 
    events are not preseent:
    
    - SENT
    - OPENED
    - CLICKED
    
    This function generates a record of Events occurred for each campaign, and then checks if the events listed
    above are in that record. Campaigns each of the events above are not observed are removed from the data.
    
    ACCEPTS:
    One dataframe: 
     - Email Data
     
    RETURNS:
    One dataframe:
    - Email Data with the campigns that do not have SENT, OPENED, and CLICKED are filtered out.
    '''
    
    # Define Campaigns
    cmpgns = df.select('CMPGN_NM').distinct().toPandas()
    cmpgns = cmpgns['CMPGN_NM'].tolist()
    
    # Key Events Dataframe Creation
    key_events = ['SENT', 'OPENED', 'CLICKED']
    df_key_events = spark.createDataFrame(key_events, StringType())
    df_key_events = df_key_events.withColumnRenamed("value", 'VENDOR_EVENT_TYPE_TXT')

    cmpgns_key = []

    # Identify Campaigns where each key event occurred.
    for c in cmpgns:
        events_in_cmpgn = df.filter(F.col('CMPGN_NM') == c) \
                            .filter(F.col('VENDOR_EVENT_TYPE_TXT').isin(key_events)) \
                            .select('VENDOR_EVENT_TYPE_TXT').distinct()

        df_missing_events = df_key_events.exceptAll(events_in_cmpgn)

        if df_missing_events.count() == 0:
            cmpgns_key.append(c)   
    
    # Filter Email Data to only include campaigns with each key event
    df_key = df.filter(F.col('CMPGN_NM').isin(cmpgns_key))
    
    print(f"Number of Campaigns: {len(cmpgns_key)}")
    
    return df_key


def data_processing(df_email, df_ewt, df_ewt_raw):
    '''
    The following function perfors the necessary data processing steps for the 
    Email, EWT, and EWT-Raw datasets before they are joined.
    
    ACCEPTS:
    Three dataframes: 
     - Email Data
     - EWT-Raw Data
     - EWT Data
     
     RETURNS:
    Three dataframes, processed: 
     - Email Data
     - EWT-Raw Data
     - EWT Data
    '''
    
    # Email
    ## Drop Null Client IDs
    df_email = df_email.where(F.col('CLIENT_ID').isNotNull())
    
    ## Filter to only Sent, Opened, and Clicked
    key_events = ['SENT', 'OPENED', 'CLICKED']
    df_email = df_email.where(F.col('VENDOR_EVENT_TYPE_TXT').isin(key_events))  
    
    # EWT and  EWT-Raw
    ## Reduce Data to Essential Columns
    col_ewt = ['PLY_POLICY_ID', 'PLY_POLICY_NBR' ,'PLY_PROCESS_DT','PLY_PROCESS_END_DT']
    col_ewt_raw = ['CPL_CLIENT_ID', 'CPL_POLICY_ID' ,'CPL_PROCESS_DT','CPL_PROCESS_END_DT']

    df_ewt = df_ewt.select(col_ewt)
    df_ewt_raw = df_ewt_raw.select(col_ewt_raw)
    
    ## Filter Observations When the END_DT precedes the START_DT
    df_ewt = df_ewt.filter(F.col('PLY_PROCESS_END_DT') >= F.col('PLY_PROCESS_DT'))
    df_ewt_raw = df_ewt_raw.filter(F.col('CPL_PROCESS_END_DT') >= F.col('CPL_PROCESS_DT'))
    
    ## Remove Leading Zeros from Policy Number
    df_ewt = policy_number(df_ewt, 'PLY_POLICY_NBR')
    
    return df_email, df_ewt, df_ewt_raw


def add_policy_number_to_email(df_email, df_ewt, df_ewt_raw):
    '''
    The following function adds the Policy Number to the email dataset.
    It does this by performing two left joins:
    
    1. Join Email to EWT-Raw by CLIENT_ID
    2. Join Step 1 data to EWT by POLICY_ID
    
    Both the EWT and EWT-Raw data sets are time-chained. Each observation has
    a start and end date-timestamp. Therefore, after each join, the resulting 
    dataframes are filtered to ensure the date-timestamp for each email 
    (VENDOR_REC_RECVD_DT) falls within the start and end date-timestamp for the 
    joined data.
    
    After the joins all columns no longer needed are dropped. 
    
    ACCEPTS:
    Three dataframes: 
     - Email Data
     - EWT-Raw Data
     - EWT Data
     
     RETURNS:
     One dataframe
    '''
    
    # Join 1 - Email to EWT-Raw by CLIENT_ID/CPL_CLIENT_ID
    condition_join_1 = [df_email.CLIENT_ID == df_ewt_raw.CPL_CLIENT_ID]

    df_join_1 = df_email.join(
                    other=df_ewt_raw, 
                    on=condition_join_1, 
                    how="left")

    # Filter Time-Chain
    df_join_1 = df_join_1.where(F.col('VENDOR_REC_RECVD_DT') \
                                .between(F.col('CPL_PROCESS_DT'), F.col('CPL_PROCESS_END_DT')))
    
    # Join 2 - Join 1 Data to EWT by POLICY_ID / PLY_POLICY_ID
    condition_join_2 = [df_join_1.CPL_POLICY_ID == df_ewt.PLY_POLICY_ID]

    df_join_2 = df_join_1.join(
                    other=df_ewt, 
                    on=condition_join_2, 
                    how="left")

    # Filter Time-Chain
    df_join_2 = df_join_2.where(F.col('VENDOR_REC_RECVD_DT') \
                                .between(F.col('PLY_PROCESS_DT'), F.col('PLY_PROCESS_END_DT')))
    
    # Drop Columns
    columns_drop = ['CPL_CLIENT_ID', 'CPL_POLICY_ID', 'CPL_PROCESS_DT', 'CPL_PROCESS_END_DT',
                    'POLICY_ID', 'PLY_PROCESS_DT', 'PLY_PROCESS_END_DT', 'HOUSEHOLD_ID']
    
    df_email_policy_num = df_join_2.drop(*columns_drop)  
    
    
    return df_email_policy_num

def filter_duplicate_emails(spark_df_email):
    '''
    The following function identifies SENT emails which may be "duplicates",
    and then filters the data so only one "duplicate" is maintained in the
    data. 
    
    An email is identified as a "duplicate" if it meets the following criteria:
    
    - VENDOR_EVENT_TYPE_TXT = SENT
    - Policy Number is the same
    - Campaign is the same
    - VENDOR_EVENT_DT is the same
    - EMAIL_ID is different
    
    INPUT:
    One dataframe: 
     - Email Data
     
     RETURN:
     One dataframe:
     - Email Data
    '''

    # Add INDEX
    spark_df_email = spark_df_email.withColumn("INDEX",  F.monotonically_increasing_id())
    
    # Select Essential Columns and Filter to SENT events
    select_col = ['PLY_POLICY_NBR', 'EMAIL_ID', 'CMPGN_NM', 'VENDOR_EVENT_TYPE_TXT', 'VENDOR_REC_RECVD_DT', 'INDEX']
    
    # Drop "Duplicates"
    list_drop_dup = ['PLY_POLICY_NBR', 'CMPGN_NM', 'VENDOR_EVENT_TYPE_TXT', 'VENDOR_REC_RECVD_DT']

    spark_df_email_dup_drop = spark_df_email.dropDuplicates(subset = list_drop_dup)
    
    # Filter By Event
    spark_df_email_dup_drop = spark_df_email.select(select_col) \
                                         .where(F.col('VENDOR_EVENT_TYPE_TXT') == 'SENT')

    # Filter Email Data with Semi-Join 
    spark_df_email = spark_df_email.join(how="leftsemi", 
                                         on='INDEX',
                                         other=spark_df_email_dup_drop.select('INDEX')) \
                                   .drop('INDEX')
    
    return spark_df_email


def wrapper_add_policy_number_to_email_data(spark_df_email, spark_df_ewt, spark_df_ewt_raw):
    '''
    The following function is a wrapper function for generating following features:
        
        - Attached policy number to email data.
    
    INPUT:
    Three spark dataframes: 
     - Email data
     - EWT data
     - EWT Raw data
     
    RETURN: 
    One spark dataframe
    - Email data with policy number added
    '''
    print("Executing Function: create_email_id")
    spark_df_email = create_email_id(spark_df_email)
    
    print("Executing Function: campaign_event_filter")
    spark_df_email = campaign_event_filter(spark_df_email)
    
    print("Executing Function: data_processing")
    spark_df_email, spark_df_ewt, spark_df_ewt_raw = data_processing(spark_df_email, spark_df_ewt, spark_df_ewt_raw)
    
    print("Executing Function: add_policy_number_to_email")
    df_email_policy_num = add_policy_number_to_email(spark_df_email, spark_df_ewt, spark_df_ewt_raw)
    
    print("Executing Function: filter_duplicate_emails")
    df_email_policy_num = filter_duplicate_emails(df_email_policy_num)
    
    return df_email_policy_num


##### Feature Generation #####
def join_email_and_ir_model_data(spark_df_ir_model, spark_df_email):
    '''
    The following function joins the Email data and Intel Routing model data.
    
    ACCEPTS:
    Two spark dataframes: 
     - Email Data
     - IntelRouting Data
     
     RETURNS:
    One spark dataframe, processed: 
     - IntelRouting with Email Data
    '''
    #Join Email with Intelligent Routing Data
    spark_df_model_email = spark_df_ir_model.join(how = 'left',
                                                  on = spark_df_ir_model.CGA_SRC_POL_NBR_TXT == spark_df_email.PLY_POLICY_NBR,
                                                  other = spark_df_email) \
                                            .drop('PLY_POLICY_NBR')
    
    # Reduce to Essential Columns
    select_columns = ['CGA_ADW_EDU_ID', 'EMAIL_ID', 'CMPGN_NM', 'VENDOR_EVENT_TYPE_TXT', 
                      'VENDOR_EVENT_DT', 'CGA_CREATE_S_DT']
    
    spark_df_model_email = spark_df_model_email.select(select_columns)

    
    return spark_df_model_email

def calculate_hrs_btw_event_and_call(spark_df_model_email):
    '''
    The following function calculates the number of hours between an email event and a customer call.
    
    ACCEPTS:
    One spark dataframes: 
     - Intel Routing with Email Data
     
     RETURNS:
    One spark dataframe, processed: 
     - Intel Routing with Email Data, with Calculation
     '''
    
    # Calculate HRS Between Event and Call
    sec_per_hr = 60 * 60 

    # Calculate time between event and call
    spark_df_model_email = spark_df_model_email \
                                 .withColumn('HRS_EVENT_TO_CALL', (F.col('CGA_CREATE_S_DT').cast(LongType()) - F.col('VENDOR_EVENT_DT')/1e3) / sec_per_hr)
    
    return spark_df_model_email

def filter_to_select_window(spark_df_model_email, window_hrs):
    '''
    The following function filters a provided spark dataframe to within a set window. 
    The provided spark dataframe must contain the column 'HRS_EVENT_TO_CALL'. The 
    provided time window must be a number, not time-date value, and expressed 
    in hours.
    
    ACCEPTS:
     - spark_df_model_email: Dataframe with column HRS_EVENT_TO_CALL
     - window_hrs: Number expressing hours. 
     
    RETURNS:
     - One spark dataframe, filtered by window_hrs
     '''
    
    # Filter to Select Time Window (HRS)
    spark_df_model_email = spark_df_model_email.where(F.col('HRS_EVENT_TO_CALL').between(0, window_hrs))
    
    return spark_df_model_email


def reduce_to_most_recent_email_per_call(spark_df_model_email):
    '''
    The following function selects the most recent email event associated with each 
    unique email / email event pair. (i.e.: email_001 / SENT, email_001 / OPENED). The
    input spark dataframe must have the columns:
      - EMAIL_ID,
      - VENDOR_EVENT_TYPE_TXT
      - VENDOR_EVENT_DT
    
    ACCEPTS:
     - spark dataframe
     
    RETURNS:
    - spark dataframe
    '''
    
    spark_df_model_email_recent_event = spark_df_model_email.groupBy('EMAIL_ID', 'VENDOR_EVENT_TYPE_TXT') \
                                                            .agg(F.max('VENDOR_EVENT_DT').alias('VENDOR_EVENT_DT'))
    
    spark_df_model_email = spark_df_model_email.join(how = 'leftsemi',
                                                     on = spark_df_model_email_recent_event.columns,
                                                     other = spark_df_model_email_recent_event)

    return spark_df_model_email

def select_single_same_time_email_event(spark_df_model_email):
    '''
    The following function selects a single instance of a email event (SENT, OPENED, CLICKED)
    when multiple instances are associated with each unique call ID. This would occur
    when multiple of the same email event, associated with the same call ID, happened at
    the exact same time. This function indiscriminently filters to only grab the first instance 
    of this. 
    
    The input spark dataframe must have the following columns:
    
      - CGA_ADW_EDU_ID,
      - VENDOR_EVENT_TYPE_TXT
    
    ACCEPTS:
     - spark dataframe
     
    RETURNS:
    - spark dataframe
    '''
    
    # Create Index
    spark_df_model_email = spark_df_model_email.withColumn('INDEX_TEMP', F.monotonically_increasing_id())
    
    # Select List of Email Events w/o Multiples
    most_recent_email_rm_mult = spark_df_model_email.select('CGA_ADW_EDU_ID', 'VENDOR_EVENT_TYPE_TXT', 'INDEX_TEMP') \
                                                    .groupBy('CGA_ADW_EDU_ID', 'VENDOR_EVENT_TYPE_TXT') \
                                                    .agg(F.first('INDEX_TEMP').alias('INDEX_TEMP'))
    # Filter Multiples Out of Data
    spark_df_model_email = spark_df_model_email.join(how = 'leftsemi',
                                                     on = most_recent_email_rm_mult.columns,
                                                     other = most_recent_email_rm_mult)
    
    return spark_df_model_email

def create_cmpgn_event_pivot_col(spark_df_model_email):
    '''
    The following function generates a new column named CMPGN_NM_EVENT. This column is 
    the concatenation of the Campaign Name and Email for each row. 
    
    The input spark dataframe must have the following columns:
    
      - CMPGN_NM,
      - VENDOR_EVENT_TYPE_TXT
      
    ACCEPTS:
     - spark dataframe
     
    RETURNS:
    - spark dataframe
    '''
    
    # Start of String Up To Third Occurance of "_"
    regex_str = '^(?:[^_]*\_){2}([^_]*)'

    # IDX = 0 to grab entire string
    idx = 0

    spark_df_model_email = spark_df_model_email.withColumn('CMPGN_NM_REG', F.regexp_extract(F.col('CMPGN_NM'), regex_str, idx)) \
                                               .withColumn('CMPGN_NM_EVENT', F.concat_ws("_", F.col('CMPGN_NM_REG'), F.col('VENDOR_EVENT_TYPE_TXT'))) \
                                               .drop('CMPGN_NM_REG')
    
    return spark_df_model_email

def create_cmpgn_event_feature(spark_df_model_email):
    '''
    The following function pivots the provided spark dataframe on the column
    CMPGN_NM_EVENT, while groupby is applied to CGA_ADW_EDU_ID. This creates a 
    dataframe with CGA_ADW_EDU_ID as a column, and then the remaining columns
    being all values found in column CMPGN_NM_EVENT. The values in the pivoted
    columns are HRS_EVENT_TO_CALL.
    
    The input spark dataframe must have the following columns:
    
      - CGA_ADW_EDU_ID
      - CMPGN_NM_EVENT
      - HRS_EVENT_TO_CALL
      
    ACCEPTS:
     - spark dataframe
     
    RETURNS:
    - spark dataframe
    '''

    spark_df_cmpgn_event = spark_df_model_email.select('CGA_ADW_EDU_ID', 'CMPGN_NM_EVENT', 'HRS_EVENT_TO_CALL') \
                                               .groupBy('CGA_ADW_EDU_ID') \
                                               .pivot('CMPGN_NM_EVENT') \
                                               .agg(F.first('HRS_EVENT_TO_CALL'))
    
    return spark_df_cmpgn_event

def generate_event_count_all_events(spark_df_model_email):
    '''
    The following function generates the Event Count Feature. It generates a column for each email 
    event (SENT, OPENED, CLICKED) with the total number of each of the respective events, per
    unique call.
    
    The input spark dataframe must have the following columns:
    
      - CGA_ADW_EDU_ID
      - VENDOR_EVENT_TYPE_TXT
      - HRS_EVENT_TO_CALL
      
    ACCEPTS:
     - spark dataframe
     
    RETURNS:
    - spark dataframe
    '''
    
    # Event Count per Call
    event_count_per_call = spark_df_model_email.groupby('CGA_ADW_EDU_ID', 'VENDOR_EVENT_TYPE_TXT').count().orderBy('count', ascending = False)
    
    # Pivot Event Count to Feature Columns 
    spark_df_event_count = event_count_per_call.withColumn('FEATURE_EVENT_COUNT', F.concat(F.lit("EVENT_COUNT_"), F.col('VENDOR_EVENT_TYPE_TXT'))) \
                                               .groupBy('CGA_ADW_EDU_ID') \
                                               .pivot('FEATURE_EVENT_COUNT') \
                                               .agg(F.first('count'))
    
    return spark_df_event_count

def insert_zero_for_select_null_counts(spark_df_event_count):
    '''
    After counting the number of events with the function generate_event_count_all_events
    there are some cases where a null value was returned, but the count can be 
    accurately modeled as 0.
    
    In cases where there are no CLICKED or OPENED events with the time wind, AND 
    there are no SENT events, this function will leave the CLICKED or COUNT value
    as null. But, in cases where a CLICKED or OPENED event count is null, but either
    the SENT or other event count is NOT NULL, this means an email was available
    to the customer within the observed window. In these cases, instead of recording
    null for the event count, 0 is recorded.
    
    Input:
        - Spark Dataframe generated by generate_event_count_all_events function.
        
    Output
        - Spark Dataframe, which replaced nulls with zeros in select cases.
    
    '''
    clicked_logical_test = (F.col('EVENT_COUNT_CLICKED').isNull() & \
                                                (F.col('EVENT_COUNT_OPENED').isNotNull() | F.col('EVENT_COUNT_SENT').isNotNull()))
    
    opened_logical_test = (F.col('EVENT_COUNT_OPENED').isNull() & \
                                                (F.col('EVENT_COUNT_CLICKED').isNotNull() | F.col('EVENT_COUNT_SENT').isNotNull()))
    
    spark_df_event_count = spark_df_event_count.withColumn('EVENT_COUNT_CLICKED',
                                                           F.when(clicked_logical_test, 0).otherwise(F.col('EVENT_COUNT_CLICKED'))) \
                                               .withColumn('EVENT_COUNT_OPENED',
                                                           F.when(opened_logical_test, 0).otherwise(F.col('EVENT_COUNT_OPENED'))) \
                                               .drop('EVENT_COUNT_SENT')

    
    return spark_df_event_count

def wrapper_feature_hrs_btw_event_and_call(spark_df_model, spark_df_email):
    '''
    The following function is a wrapper function for generating following features:
        
        - Number of hours between email event and customer call for each campaign
          and event pair.
    
    INPUT:
    Two spark dataframes: 
     - Email data
     - Intellgence Routing model data
     
    RETURN: 
    One spark dataframe
    - Feature data
    '''
    print("Executing Function: join_email_and_ir_model_data")
    spark_df_model_email = join_email_and_ir_model_data(spark_df_model, spark_df_email)
    
    print("Executing Function: calculate_hrs_btw_event_and_call")
    spark_df_model_email = calculate_hrs_btw_event_and_call(spark_df_model_email)
    
    print("Executing Function: filter_to_select_window")
    # 10 Day Window
    window_hrs = 24 * 10
    spark_df_model_email = filter_to_select_window(spark_df_model_email, window_hrs)
    
    print("Executing Function: campaign_event_filter")
    '''
    This function is executed again because after the window filter, some campaigns
    may no longer have occurances of all three email event type.
    '''
    spark_df_model_email = campaign_event_filter(spark_df_model_email)
    
    print("Executing Function: reduce_to_most_recent_email_per_call")
    spark_df_model_email = reduce_to_most_recent_email_per_call(spark_df_model_email)
    
    print("Executing Function: select_single_same_time_email_event")
    spark_df_model_email = select_single_same_time_email_event(spark_df_model_email)
    
    print("Executing Function: create_cmpgn_event_pivot_col")
    spark_df_model_email = create_cmpgn_event_pivot_col(spark_df_model_email)
    
    print("Executing Function: create_cmpgn_event_feature")
    df_cmpgn_event = create_cmpgn_event_feature(spark_df_model_email)
    
    return df_cmpgn_event


def wrapper_feature_event_count(spark_df_model, spark_df_email):
    '''
    The following function is a wrapper function for generating following features:
        
        - Count of event CLICKED or OPENED executed within observation window
          before cusomter call.
    
    INPUT:
    Two spark dataframes: 
     - Email data
     - Intellgence Routing model data
     
    RETURN: 
    One spark dataframe
    - Feature data
    '''
    print("Executing Function: join_email_and_ir_model_data")
    spark_df_model_email = join_email_and_ir_model_data(spark_df_model, spark_df_email)
    
    print("Executing Function: calculate_hrs_btw_event_and_call")
    spark_df_model_email = calculate_hrs_btw_event_and_call(spark_df_model_email)
    
    print("Executing Function: filter_to_select_window")
    # 10 Day Window
    window_hrs = 24 * 10
    spark_df_model_email = filter_to_select_window(spark_df_model_email, window_hrs)
    
    print("Executing Function: generate_event_count_all_events")
    spark_df_event_count = generate_event_count_all_events(spark_df_model_email)
    
    print("Executing Function: insert_zero_for_select_null_counts")
    spark_df_event_count = insert_zero_for_select_null_counts(spark_df_event_count)
    
    return spark_df_event_count

def join_email_features_to_model_data(spark_df_model, dict_spark_feature_df):
    '''
    The purpose of this function is to join the email feature dataframes to the 
    exisiting model data. Feature dataframes must be stored in a dictionary. This
    allows for this function to process any number of feature dataframes that may
    be created.
    
    All Spark DataFrames stored as values in the dictionary must have the following column:
    
      - CGA_ADW_EDU_ID
    
    INPUT:
     - spark_df_model: Spark Dataframe of model data
     - Dictionary: Values must be Spark Dataframes of the features
     
    RETURN: 
    One spark dataframe
    - Model data with new features
    '''
    
    for spark_df_feature in dict_spark_feature_df.values():
        
        spark_df_model = spark_df_model.join(how='left',
                                             on = 'CGA_ADW_EDU_ID',
                                             other = spark_df_feature)
    
    return spark_df_model
