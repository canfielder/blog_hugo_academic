---
title: Virtualization and Containers - Part 2
author: Evan Canfield
date: '2020-02-09'
slug: dsba-6190-discussion-week-4-virtualization-and-containers-part-2
categories:
  - discussion
  - cloud
tags:
subtitle: 'DSBA-6190 Discussion: Week 4'
summary: 'Week 4 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# What are containers?
Containers provide a complete isolated environment on a virtual machine. This is functionally a different form of virtualization. Consider a cloud computing environment. The backbone of the cloud computing world are physical server farms hosting everything. Within these servers are the virtualization concept of Virtual Machines (VM). VMs allow hardware (the server farms) to host multiple operating systems on a single piece of hardware. Containers are just this virtualization concept taken one layer deeper. Within each VM, numerous containers can be deployed, each container splitting the VM space into virtual compartments, isolated from one another.

# What problem do containers solve?
Contains solve the problem of portability. If an app in a Docker container is stable, it is easy to move that Docker container to another host OS as long as that OS has the Docker server-side daemon available. This way you know whatever is in the container will run anywhere. You don’t have to worry about creating an app using Python 2, moving it to a different environment, and realizing later the new environment runs on Python 3.

Another problem containers address is optimizing the use of hardware’s computing resources. VMs allow multiple OS systems to run on a piece of hardware, utilizing space that might otherwise be unused. Containers push this concept to another layer. If a VM has computing resources not being used, another properly sized container could be added to it. No need to spin up a whole other VM to try and take advantage.

# What is the relationship between Kubernetes and containers?
Kubernetes is a platform for orchestrating the deployment of containers. Two of the major components of a Kubernetes system are nodes and pods. A node can be thought of as a server equivalent. A pod is a collection of one or more containers. In the Kubernetes system the nodes run the pods. A pod may contain a container which has a single app, or it may also contain “side-car” containers. The side-car containers are containers which support the functionality of the main app, but don’t have any purpose on their own.

In order to be robust enough to be used in a production environment, an application will often use more than one pod. Multiple pods of the same app provide fault tolerance. If one pod goes down, another can quickly fill in, performing the same function. In order to do this seamlessly, there needs to be some control orchestrating the pod deployment. This is what the Kubernetes system provides. It is able to orchestrate the interaction between all of the pods and nodes. 

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This weeks training was the Google QwikLabs on **Orchestrating the Cloud with Kubernetes**. I made it through without much issue, I can tell implementing Kubernetes on my own is going to be a lot of “two steps forwards, one step back”. It looks to be a very powerful tool. The part of the training on secure services took me a while to wrap my head around.

{{< figure src="imgs/kubernetes.png" lightbox="true" width="95%">}}


