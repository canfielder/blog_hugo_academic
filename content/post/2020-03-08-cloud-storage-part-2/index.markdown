---
title: Cloud Storage - Part 2
author: Evan Canfield
date: '2020-03-08'
slug: dsba-6190-discussion-week-8-cloud-storage-part-2
categories:
  - discussion
  - cloud
tags:
  - GCP
subtitle: 'DSBA-6190 Discussion: Week 8'
summary: 'Week 8 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# What are the key differences between block and object storage?
Block storage has a structure that, to me, feels very rational. Storage space is broken up into evenly sized blocks of data. There is no metadata, no description or context, applied to each block. Instead the block is just a chuck of data. This schema is very useful for structured databases, particularly databases cataloging transactions.

Object storage, on the other hand, doesn’t split data into raw blocks. The storage is based on the metadata, the context. Objects tagged by a particular piece of metadata are stored in an object, but that object isn’t a uniform size. The object is the size it needs to be depending on what is stored in the object. This makes object storage very powerful for unstructured data.


# What are the key problems that a Data Lake solves?
Data Lakes provide an alternative to other, more rigid, storage structures. Before the Data Lake, when storing large amounts of data, you were generally dealing with a database, or maybe a data warehouse. The databases would usually handle structured data. More recently unstructured data was being stored in structures like MongoDB. These databases were excellent at the narrowly defined tasks they were set up for.

The main property that allows for this is called **schema-on-read**. What this means, is a structure is not imposed on the data when it’s place in the lake. The structure comes when data is read and transformed from the lake, into your workflow progress. Relational databases are **schema-on-write**, which means you need a structure in place before you put the data in the database. This is limiting if you’re not sure what parts of the data you want to investigate, and hence don’t know how to structure a database. Now, with the Data Lake, all the data available.

This new form with increased flexibility also means increased difficulties. Without proper management a Data Lake can become a Data Swamp, and that beautiful interplay between all your different data sources is now indecipherable. Also, understanding the Data Lake takes more expert abilities. Previously, once a relational database was constructed (which does require expertise), it was actually fairly clear how to it worked, and how everything is related. Now a level of expertise is needed in knowing how to interact with the Data Lake and transform the data you need into a useful structure.

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This week’s training was a Google Qwiklab about using the Cloud SDK command line. Pretty straightforward stuff. There was a feature for gcloud called interactive (it's in beta). This feature helps auto-fill command line arguments and presents prompts on what arguments are required. Very similar to working out of a notebook. I found it extremely useful and hope it gets further developed.

{{< figure src="imgs/gcloud_beta_interactive.png" lightbox="true" width="95%">}}


