# Blocks World -- Simulator, Code, and Models (Misra et al. EMNLP 2017)

[Mapping Instructions and Visual Observations to Actions with Reinforcement Learning](https://arxiv.org/abs/1704.08795)  
Dipendra Misra, John Langford, and Yoav Artzi  
In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2017.  

<p align="center"><a href="https://youtu.be/fmCP-SdDOT0"><img src="http://yoavartzi.com/temp/emnlp2017-video.gif"></a></p>

The original environment was designed by [Bisk et al. 2016](http://yonatanbisk.com/papers/2016-NAACL.pdf), who also collected the [data](https://nlg.isi.edu/language-grounding/). 

## Run the Code in 60 Seconds (Requires only python 2.7)

In this section, we will run the oracle baseline on the devset. 
This will give an idea of the simulator and the code and does not requires any dependency besides python 2.7.

**Supports:** Mac OS and Linux

**Requires:** python2.7

### Running on Mac

1) Clone the code  ``git clone https://github.com/clic-lab/blocks``
2) Go to ``./blocks/BlockWorldSimulator/`` and run ``build_mac.app`` by double clicking.

   Choose the Fastest mode setting and any resolution (does not matter which resolution).
   
3) Now run the oracle baseline by running the following command in the home directory.
   May take 5-10 seconds for the simulator to be ready and before following command works.
     
      ``cd ./blocks/BlockWorldRoboticAgent/``
      
      ``python agent_oracle.py``
      
4) The log will be generated in ``./blocks/BlockWorldRoboticAgent/log.txt`` and final number should match
    the numbers in the paper *0.35 mean distance error*.

### Running on Linux

1) Clone the code  ``git clone https://github.com/clic-lab/blocks``
2) Go to ``./blocks/BlockWorldSimulator/`` and make the file ``linux_build.x86_64`` executable by running:

     ``chmod 777 linux_build.x86_64``

3) Now run the file ``linux_build.x86_64`` by double clicking.

   Choose the Fastest mode setting and any resolution (does not matter which resolution).
   
3) Finally run the oracle baseline by running the following command in the home directory.
   May take 5-10 seconds for the simulator to be ready and before following command works.
     
      ``cd ./blocks/BlockWorldRoboticAgent/``
      
      ``python agent_oracle.py``
      
4) The log will be generated in ``./blocks/BlockWorldRoboticAgent/log.txt`` and final number should match
    the numbers in the paper *0.35 mean distance error*.
    
Instructions for running other baselines will come soon.

-------

**Production Release and pre-trained models coming soon ...**
