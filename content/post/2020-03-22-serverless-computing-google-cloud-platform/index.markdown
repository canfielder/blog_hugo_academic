---
title: Serverless Computing - Google Cloud Platform
author: Evan Canfield
date: '2020-03-22'
slug: dsba-6190-discussion-week-10-serverless
categories:
  - discussion
  - cloud
tags:
  - GCP
subtitle: 'DSBA-6190 Discussion: Week 10'
summary: 'Week 10 Discussion for DSBA 6190 - Introduction to Cloud Computing'
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

# What problems does Google App Engine solve?
The Google App Engine (GAE) solves similar problems to other cloud-based services I have covered in previous weeks. Namely, it shifts system management duties to outside experts which allows the user to focus their efforts on their product. GAE is a PaaS, Platform as a Service. That means Google engineers will spend their time (and considerable expertise) handling platform and infrastructure management. If your team, or you personally, don’t have those skills, this is a huge win. Not only do you not lose time haphazardly building those skills, there are actual experts who are going to pick that slack for you.

Using GAE means Google manages system operations. These operations cover a wide range of features, including scaling, A/B testing, fault tolerance, and host server management. Now this also puts you in the Google environment (obviously). On the plus side, this means easier access to very powerful built-in APIs to a wide variety of interesting tools. I’m partial to some of Google’s Natural Language Processing. With a simple call to the API you’re able to quickly do Of course, being in the Google environment may also be constricting, if there are actions you want to take the Google doesn’t support, yet. This is certainly a trade-off worth considering.

Like previous software services we’ve covered in this class, GAE is a pay as you go service. This means only paying for what you use. For example, when taking advantage of GAE scaling capabilities, you don’t have to pay for a system that is always running at peak capacity. You don’t have to size for worst case scenarios, pay peak prices during down times. Instead, you pay for what you use.

## Resources
Thank you to the following resources for help and input on this question.

* [https://www.netsolutions.com/insights/what-is-google-app-engine-its-advantages-and-how-it-can-benefit-your-business/](https://www.netsolutions.com/insights/what-is-google-app-engine-its-advantages-and-how-it-can-benefit-your-business/)
* [https://blog.realkinetic.com/why-google-app-engine-9c3d2f75dd02](https://blog.realkinetic.com/why-google-app-engine-9c3d2f75dd02)

# What problems does the Cloud Shell environment solve?
The Cloud Shell is a command line program in the Google Cloud Platform. Its utility is most evident in how quick and easy it is to be up and running in the shell. Just a quick click from the Cloud Console and you’re in. From here you can access almost any resource that you want, so long as you have the right authorizations.

One of my favorite features in the Cloud Shell is the available IDE. With the IDE you can use the Shell while at the same time viewing code files and file structure. The IDE makes it very easy to inspect your code and edit files. Another one of my favorite features is the build in web preview button. This really helps when debugging a flask app or active docker container. Just one click and you can see your deployed product.

To top it off the feature is free. So, while Google Cloud Shell has some nifty features, I don’t think there is anything you technically need the shell for. With the Google Console interface, and other Google tools, you could probably get by without it. But the shells utility and integration so much else in the Google environment, along with price, make it an essential tool.

## Resources
Thank you to the following resources for help and input on this question.

* [https://medium.com/google-cloud/getting-the-best-out-of-google-cloud-shell-3d6ca64bc741](https://medium.com/google-cloud/getting-the-best-out-of-google-cloud-shell-3d6ca64bc741)

# Interactive Lab
### Post a screenshot of a lab where you had difficulty with a concept or learned something.
This week’s training was about accessing and using Virtual Machines within Google Cloud Platform, on Google Qwiklabs. This training introduced me to another environment I could potentially work and develop in. This appears to be a less structured sandbox than other environments, which has its pros and cons. It’s good to know if I need more flexibility these resources are available.

No screenshot this time. The training wasn’t very visual, and the one screenshot I took was badly sized, cropping some necessary information.


