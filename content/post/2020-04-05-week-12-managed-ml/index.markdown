---
title: 'Managed ML Systems'
author: Evan Canfield
date: '2020-04-05'
slug: dsba-6190-discussion-week-12-managed-ml
categories:
  - discussion
tags:
  - AWS
subtitle: 'DSBA-6190 Discussion: Week 12'
summary: 'Week 12 Discussion for DSBA 6190 - Introduction to Cloud Computing'
authors: []
lastmod: '2020-03-29T10:17:48-04:00'
#draft: True 
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
#### A note on the series:
This post is part of a series of weekly discussion prompts for my class **DSBA-6190: Introduction to Cloud Computing**. Each prompt includes two or three general questions on a cloud computing related topic, along with screenshot from a interactive online lab. 

# What problems does Sagemaker solve?
Sagemaker is a service provided by Amazon Web Services that per Amazon, “is a modular, fully managed machine learning service that enables developers and data scientists to build, train, and deploy ML models at scale.” Sagemaker places many of the parts required to develop a production level model in one location, allowing for a more streamlined process and a clean machine learning (ML) pipeline. It also rests on a Jupyter Notebook framework, which many of today’s data scientists and machine learning users are familiar with. 

With Sagemaker, a user is able to:

* Import the data, from an internal AWS location (like S3) or from outside AWS
* Clean and transform the data
* Train a model from several libraries, including Sagemaker's own and the popular Sci-kit Learn library
* Deploy the model to an endpoint, which can then be called from anywhere at any time
* Automatically scale the endpoint, allowing for seamless access

If the user is not an expert in creating and tuning ML models, Sagemaker also has autopilot capabilities. Using this function, the user does not have to be an expert coder or data scientist. It removes the need to enter all the necessary information in a traditional code script or notebook method. If you are not familiar with Python or another ML framework, trying to parse that code can be very intimidating. With Sagemaker Autopilot, all the user needs to do is provide the tabular data, and identify a column to predict. This works for both regression and classification problems. 

<figure>
  <img src="imgs/sagemaker_autopilot.png" lightbox="true" width="95%">
</figure> 

[**Source**](https://aws.amazon.com/sagemaker/autopilot/)
  
There are other automatic machine learning tools on the market, outside of Amazon, but Amazon bills this tool as less of a black box when compared to other AutoML competitors.  Autopilot does all of the heavy lifting, while providing greater visibility, and therefore confidence, in the generated models.

## Resources
Thank you to the following resources for help and input on this question.

* [**Amazon SageMaker Autopilot**](https://aws.amazon.com/sagemaker/autopilot/)
* [**How AWS Attempts To Bring Transparency To AutoML Through Amazon SageMaker Autopilot**](https://www.forbes.com/sites/janakirammsv/2020/02/27/how-aws-attempts-to-bring-transparency-to-automl-through-amazon-sagemaker-autopilot/#3b1092b32acc)

# What are competitive offerings to Sagemaker?

The competitors to Sagemaker all fall within the Machine Learning as a Service (MLaaS) bucket. They all offer, at some level, the ability to develop and deploy machine learning models in one place. Some competitors also offer versions of automatic ML (AutoML). The following chart shows a high level overview of the major competitors to Sagemaker for both AutoML and custom made models.

<figure>
  <img src="imgs/ml_framework_summary.png" lightbox="true" width="95%">
</figure> 

[**Source**](https://www.altexsoft.com/blog/datascience/comparing-machine-learning-as-a-service-amazon-microsoft-azure-google-cloud-ai-ibm-watson/)

As you can see each service offers slightly different opportunities. For example, Microsoft Azure provides the greatest amount of automatic ML model types, but does not provide its own algorithms like the other services. For custom model building, each service supports a variety of ML frameworks. 

## Resources
Thank you to the following resources for help and input on this question.

* [**Comparing Machine Learning as a Service: Amazon, Microsoft Azure, Google Cloud AI, IBM Watson**](https://www.altexsoft.com/blog/datascience/comparing-machine-learning-as-a-service-amazon-microsoft-azure-google-cloud-ai-ibm-watson/)

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This week the training was **Analyze Data with Amazon Sagemaker, Jupyter Notebooks and Bokeh** from Vocareum Academy Data Analytics. This was a pretty simple walkthrough. Very useful for learning how to set up an instance, and also useful for learning some basics with Bokeh. I haven’t used Bokeh before, I’m more familiar with R visual tools like ggplot, but the visuals created looked pretty good, better than Matplotlib. 

{{< figure src="imgs/sagemaker_bokah.png" lightbox="true" width="95%">}}

I also worked through the Sagemaker tutorial from Amazon Web Services, [Build, Train, and Deploy a Machine Learning Model](https://aws.amazon.com/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/). The ability to quickly train, deploy, and evaluate a model was really impressive. This is definitely an attractive service for getting ML models into production.

{{< figure src="imgs/sagemaker_deploy_predict_evaluate.png" lightbox="true" width="95%">}}
