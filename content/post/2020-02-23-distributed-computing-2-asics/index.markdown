---
title: 'Distributed Systems - Part 2'
author: Evan Canfield
date: '2020-02-23'
slug: dsba-6190-discussion-week-6-distributed-computing-2-asics
categories:
  - discussion
tags:
  - GCP
subtitle: 'DSBA-6190 Discussion: Week 6'
summary: 'Week 6 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# How could ASICs play an important role in Machine Learning going forward?
Application Specific Integrated Circuits (ASICs) are hardware circuits designed for a specific function. In order to understand how they may play a role in Machine Learning, I think it’s important to understand the other the other hardware options for running machine learning algorithms.

## General Purpose Processing
Today, most computers run on Central Processing Units (CPUs). CPUs are general purpose processor, meaning the same CPU can be used for many of different operations. If you’re running a machine learning algorithm on your laptop, say a Random Forest classification via Python, it’s running through your computer’s CPU. But so is your email, and your word processing software. Unless you have a unique setup, any software run on a run-of-the-mill computer is run through the CPU.

Graphic Processing Units (GPUs) are another common processor. GPUs, like the name implies, were developed to run graphic based processes, like high performance video gaming. But they also work well for machine learning algorithms. GPUs are like CPUs but each processor has thousands of tiny multipliers. This is very useful for massively parallel tasks, like rendering visuals (as was intended), or running a machine learning algorithm with complex matrixed operations (as we later learned worked well).

## Task Specific Processing
The purpose of ASICs is to go against the general purpose use of CPUs and GPUs. An ASIC is developed specifically for a task. As an example, one ASIC is the Tensor Processing Unit (TPU) developed by Google. TPUs were developed to run one specific type of task: large matrix operations. With this specified design, TPUs are able to process large matrix operations faster, and with less memory access, than a similar CPU or GPU system.

## The Future
With the use of machine learning techniques expanding, and being applied to growing amounts of data, it’s easy to see how increased speed is important. But why go through the expensive process of developing brand new component? Conventional computers are becoming cheaper, so why not just buy more and more conventional CPU and GPU based devices. It seems to work for general cloud computing.

The reason generally comes down to a mix of Amdahl’s Law and the end of Moore’s Law. Moore’s Law was the idea that computing power would double every 18 months. Those conditions haven’t been seen since the early 2000’s and currently process power is only increasing at a rate of doubling every 20 years (3% increase per year).

{{< figure src="imgs/moores_law_cropped.png" lightbox="true" width="95%">}}

**Source:** [ https://riscv.org/wp-content/uploads/2017/05/Mon0915-RISC-V-50-Years-Computer-Arch.pdf]( https://riscv.org/wp-content/uploads/2017/05/Mon0915-RISC-V-50-Years-Computer-Arch.pdf)

Amdahl’s law regards the relationship to the number of processors and the speedup of a processing. Amdahl’s Law shows additional limitation to just adding available processors to tackle bigger machine learning tasks. No matter the level of process parallelization, the effect of additional processors plateaus.

{{< figure src="imgs/amdahl.png" lightbox="true" width="95%">}}

**Source:** [https://upload.wikimedia.org/wikipedia/commons/e/ea/AmdahlsLaw.svg](https://upload.wikimedia.org/wikipedia/commons/e/ea/AmdahlsLaw.svg)

This is why new specialized ASICs will be very important for machine learning going forward. The current general components have limited ability to handle the increasing size and complexity of machine learning tasks. These new ASICs will help handle what is to come.

I do need to point out one caveat. I think an important question is how much gain can ASICs actually provide? I did find one paper, out of Princeton, which indicated the ASICs will eventually hit their own limits ([here](http://parallel.princeton.edu/papers/wall-hpca19.pdf)). So, it is possible the growth of ASICs will lead to ever increasing speeds. Or perhaps, ASICs will just be a step-function. A quick shot in the arm to handle large quantities of data, but then speed increases will simply level off again.

# Interactive Lab
### Find, run and extend a notebook in seedbank and share along with your comments on what you learned.

I ended up playing around with the Performance RNN model. This model allowed me to create random samples of music and visually displayed the polyphonic notes generated. The model was already trained to a provided set of music, so if I wanted to really play with the model I’d want to supply the model with a new music set, and then retrain the model. So there wasn’t a ton of interaction I could do, besides adjust one hyperparameter (temperature).

One thing I realized was there wasn’t much documentation on the code that I could find. I was hoping to read up on the packages to see what each line was actually doing, but this was pretty hard to track down.


{{< figure src="imgs/random_piano.png" lightbox="true" width="95%">}}
