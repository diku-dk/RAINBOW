---
layout: sub_page
title: "GUI guide"
subtitle: "Upload Code"
role: "child"
prefix: "contribute"
postfix: "gui_guide/upload_guide"
permalink: my_collections/sub_help_pages/contribute/gui_guide/upload_guide
---
# Upload Code
1. We have coded a new simulator that we want to submit to the libRAINBOW repository for this example.  
Therefore we start by navigating to the simulator folder. Then, if you have made another implementation, such as a mathematical module, you place it in the math folder.
In the simulation folder, create a new folder with your implementation name. In this case, "new_simulator".  
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial.png)
2. Then, in your newly created folder, insert your implementation. 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_2.png)
3. Open "GitHub Desktop". Ensure that all your implementation is visible in "changed file". 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_3.png)
4. Write title use the format: < issue number >: your commit messages. Thereafter, write a short 
description. Then press "commit".
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_4.png)
5. Press "Push origin"
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_5.png)
6. Return to the GitHub webpage. Press "Compare & pull results" 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_6.png)
8. Define "pull results"
    1. Assign Reviewers
    2. Assign Assignees
    3. Assign Label
    4. Assign Issue
    5. Before merge -> All revierwers mus approve
    6. Before merge -> All tests must pass 
    7. Then press "Squash and merge" 
![]({{ site.url }}{{ site.baseurl }}/assets/images/pull_request/libRAINBOW_tutorial_8.png)
9. Next, adding unittest to your code.