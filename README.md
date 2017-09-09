# Blocks World -- Simulator, Code, and Models (Misra et al. EMNLP 2017)

[Mapping Instructions and Visual Observations to Actions with Reinforcement Learning](https://arxiv.org/abs/1704.08795)  
Dipendra Misra, John Langford, and Yoav Artzi  
In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017.  

[![demo](http://yoavartzi.com/temp/emnlp2017-video.gif "Demo")](https://youtu.be/fmCP-SdDOT0)

# Run the code in 60 seconds 

In this section, we will run the oracle baseline on the devset. This will give an idea of the simulator and code 
and does not require tensorflow.

Supports: Mac and Linux Build
Requires: python2.7

**Instructions for running on Mac**:

1) Clone the code  ``git clone https://github.com/clic-lab/blocks``
2) Go to blocks/BlockWorldSimulator/ and run build_mac.app
   Choose the Fastest mode setting and any resolution (does not matter which resolution).
3) Now run the oracle baseline. Go to BlockWorldRoboticAgent and run the following command:
     
      ``python agent_oracle.py``

**Instructions for running on Mac**:

Same as above except run linux_build.x86_64 instead of build_mac.app
You may have to give executable permission to the build. To do so run:

`` chmod 777 linux_build.x86_64``

Instructions for running other baselines will come soon.

**Production Release and pre-trained models coming soon ...**
