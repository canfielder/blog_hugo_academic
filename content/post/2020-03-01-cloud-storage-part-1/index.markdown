---
title: Cloud Storage - Part 1
author: Evan Canfield
date: '2020-03-01'
slug: dsba-6190-discussion-week-7-cloud-storage-part-1
categories:
  - discussion
  - cloud
tags: ''
subtitle: 'DSBA-6190 Discussion: Week 7'
summary: 'Week 7 Discussion for DSBA 6190 - Introduction to Cloud Computing'
authors: []
lastmod: '2020-03-29T15:57:55-04:00'
featured: no
#draft: true
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
#### A note on the series:
This post is part of a series of weekly discussion prompts for my class **DSBA-6190: Introduction to Cloud Computing**. Each prompt includes two or three general questions on a cloud computing related topic, along with screenshot from a interactive online lab.

# What are the problems with a “one size fits all” approach to relational databases?
The one-size-fits all approach to databases used to be the default position. Every database was a relational database, no matter the data you had or the use case you were applying it to. Using a relational database for the wrong application can lead to wasted and/or slow actions. For example, before Amazon built DynamoDB, they were also using standard relational databases, but pushing their limits. About 70% of their operations were key-value lookups. Other functions available in a relational database were essentially not being used. This led to the creation of other database systems, like DynamoDB. Databases built to a purpose are more useful in a world with every growing amounts of data.

# How could a service like Google BigQuery change the way you deal with data?
A service like Google BigQuery assists in handling very large amounts of data. This could be particularly useful in early EDA steps when dealing with very large datasets. Instead of importing all of the available data into a modeling environment at the start, early descriptive analysis can more quickly by done via BigQuery. This can help inform early insights much faster.

This could also expand past straight forward EDA and into modeling. Some basic modeling techniques can be employed using BigQuery ML. I’ve not dug into this too much, but I’ve seen the tutorials on logistic regression and kmeans clustering. Similar to the time gains made by running EDA in BigQuery, if you know you’re going to run some simple models on your data, it might be useful to run them in BigQuery, at least just as a test, instead of exporting and uploading to a different environment, just to run the same model.


# What problem does a “serverless” database like Athena solve?
Serverless databases, like Athena, can function simply and efficiently for the user, compared to more traditional databases. Particularly with Athena, the user does not have to setup or manage servers or data warehouses. That is handled by the database provider. All of the data used in Athena is hosted on Amazon S3. Once the data is loaded and the schema defined, the user is immediately able to run queries.

The costs of these databases are based on the queries run, not on up-front cost. You only pay for what you use. This prevents the user from overpaying for services unused, but also helps when the user needs to scale up or down. Much like other cloud computing services, you don’t need to purchase resources to cover the maximum expected usage at all times. Serverless databases allow for quick scaling.

Additionally, Serverless databases have the potential to be very fast. Athena, in particular, automatically executes queries in parallel, allowing for very fast and efficient querying.


# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
I have previous experience with BigQuery, but I’d never used the MySQL Instance functionality. This was a pretty cool way to interact with SQL directly via the command line.

{{< figure src="imgs/big_query_mysql.png" lightbox="true" width="95%">}}


