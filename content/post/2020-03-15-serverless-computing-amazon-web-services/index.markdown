---
title: Serverless Computing - Amazon Web Services
author: Evan Canfield
date: '2020-03-15'
slug: dsba-6190-discussion-week-9-serverless-aws
categories:
  - discussion
  - cloud
tags:
  - AWS
subtitle: 'DSBA-6190 Discussion: Week 9'
summary: 'Week 9 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# What are the tradeoffs with serverless architecture?
Serverless architecture presents a world where the general user does not have to worry about server management. Of course, someone out there still has to, but this responsibility is relegated to the vendors providing the serverless service. Serverless does not just save time and effort, it has other tangible benefits.

Overall costs are a large benefit for most users. With serverless, you only pay for what you use. You are not paying for idle space you are not using. This dynamic pricing also pairs well with the elasticity of serverless systems. The systems will auto-scale and provision based on need. There is no need to the user to manually adjust this.

The benefits of serverless also come with several tradeoffs. The following a few of the more important tradeoffs:

* **Vendor Lock-In:** The benefit of having a vendor handle managing the servers means you are also at the mercy of the vendor. Without inventive system engineering, you are stuck using the systems provided by the vendor.
* **Tests:** It is not always easy to replicate a serverless environment for the testing stage. This makes full testing and debugging efforts difficult in some cases.
*	**Price:** While pay-for-what-you-use pricing is cheaper in many instances, if you have long-running applications, the costs of using a serverless system could exceed a self-management environment. Make sure you know what situation you’re going to be in before you invest time and money.

# What are the advantages to developing with Cloud9?
Cloud9 provides a useful environment for development with several unique advantages:

* **Connection To Amazon System:** Cloud9 is a part of Amazon Web Services (AWS), and can therefore easily connect to other applications within AWS. API keys for each individual Amazon application are not necessary. Additionally, SDKs are already installed. For working with Python, the SDK is the Boto3 package. Using Boto3 helps facilitate connection between Amazon services. Services like S3, or Dynamo Database. Cloud9 provides a useful central location for this development.
* **Collaboration:** Users are able to share their Cloud9 workspace, facilitating collaboration with teams. This is all set up with AWS Identify and Access Management tool.
* **Cost Savings:** The Cloud9 environment provides a cost-saving backstop. Instances automatically stop after being idle for 30 minutes. This prevents an instance running unnoticed and racking up charges. Personally, this actually happened to me with a Sagemaker image. Sagemaker does not have this backstop. I left an idle instance running for a couple days, unknowingly. That ended up costing me a couple bucks.

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This week’s training was about using Lambda Triggers. The training was hosted by Vocareum. The training was very useful as I am considering using Lambda Triggers as part of my final project workflow. The primary flow will be the use of **Code Deploy** launching a Sagemaker Instance, which will train and deploy a model. I am thinking of using Lambda Triggers to provide some more flexibility in how and when this deploymen gets triggered, or if needed, rolled back.

{{< figure src="imgs/vocareum_lambda.png" lightbox="true" width="95%">}}


