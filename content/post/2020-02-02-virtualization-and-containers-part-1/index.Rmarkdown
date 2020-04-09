---
title: Virtualization and Containers - Part 1
author: Evan Canfield
date: '2020-02-02'
slug: dsba-6190-discussion-week-3-virtualization-and-containers-part-1
categories:
  - discussion
  - cloud
tags:
subtitle: 'DSBA-6190 Discussion: Week 3'
summary: 'Week 3 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# What are the different layers of network security on AWS and what unique problems do each solve?
Amazon Web Services (AWS) uses a shared responsibility model when it comes to security. Some security responsibilities are handled by Amazon, while the remaining are the responsibility of the user.

The primary area of Amazon’s responsibility is in connection to the infrastructure side of the system. The infrastructure covers everything from the more abstract virtualization layer, to the actual physical components and sites that make up the Amazon cloud. Essentially, if something is connected to keeping AWS running, then Amazon has responsibility for it.

The customer has responsibility for security of systems and files they put in the AWS cloud. The customer is responsible for how the customer’s data is secured. This covers where content is placed on Amazon Cloud servers, what services can interact with that content, who has access to the content, etc.

By splitting the responsibility between Amazon and the customer it allows the security of each component to be overseen by the expert of that component. If it’s the physical 
infrastructure, I want Amazon handling it. I wouldn’t want a random customer noticing a structural issue and trying to handle it themselves. The flip side is that if a customer is going to use Amazon Cloud services, they have to have a strong understanding of both cloud security and their own content’s security. 

In my current industry, nuclear power,  there are several tiers of security importance. Some documents are essentially public knowledge. Others require high levels of security. While these tiers are security are not unusual, how they breakdown are defined can differ greatly industry to industry. This division of responsibilities allows the subject matter experts to be responsible for their subject, and only their subject.

# What problem do AWS Spot Instances solve and how could you use them in your projects?
Spot instances allow the user to quickly and inexpensively set up a computing system. For example, if a project I was running required processing large amounts of data for a deep learning model, trying to run that on my laptop would be foolish. Instead, I can set up the code for the deep learning model beforehand on a standard computer. Then I can set up a AWS spot instance which the size, computing power, security requirements, and operating environments can be easily selected. I could then run the deep learning model on a AWS Spot instance that can handle the needed computing power.

Another upside for AWS spot instances is I can select an operating system configuration that will work for my needs. For example, I could select a system with Tensorflow, or Keras, or just Python, and as the AWS Spot instance spins up the correct libraries and configurations are loaded. This automation saves me time, but also covers a step that I may not be an expert in. 

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This weeks training I was unable to get the module to work. I'll try to come back to this later.

