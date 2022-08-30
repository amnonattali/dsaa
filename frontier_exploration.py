'''
i guess first I need to read the literature and understand if this has been done before
    --> honestly any count based model is just doing density estimation right??
    which means all im really doing is implementing a related work which is GOOD

Frontier:
- explore the environment (not for too long)
- use the resulting data to train a normalizing flow (or any generative model with explicit probability)
- use the model to explore with a reward proportional to the inverse probabiltiy of seeing a datapoint under the model

Related work:
- pseudocount based exploration?
-




---------------
TODO:
- map out what needs to be done to complete the paper
    - how should we frame the contribution
        -- i like the idea of framing it as "given a dataset" and then finding interesting ways to generate that data
    - what should be added or changed to make the contribution better
    - what experiments would best highlight the contribution
    - for each image how could it be improved/changed 
- map out what to do next for research
    - what is the problem we are solving

Problems we had before:
- how do we generalize to states not seen before
    - the representation learning perspective is that we need to identify features which generalize,
    meaning we need to find a mapping from the input features to a latent space such that states which have not
    yet been seen are still mapped to an area of the space which is understood based on the previous data
    - discount factor gives another perspective - things that are nearby are more important than things which are
    far away, so in other words generalization comes also from locality
        - what is locality? 
            it can mean looking at things relative to self (i.e. s'-s?)
            it can mean ignoring features that are far away? --> this is weird because it is dependent on the representation
    - I like the idea of trying to fit a single model onto different regions of space, as in saying
    "the relationship between these states here, is similar to the relationship between those states there"
    and so fitting the same abstraction onto those regions of space


- how do we use the abstraction we've learned to improve exploration

--------------
ideas:
- feature local SR
    - map x to s
    - ignore some of the features (information?) in s to get s_local
        - this means that different states have the same s_local
    - compute the SR of s_local --> meaning how does s_local change starting at x?




Possible things to code:
- frontier exploration
    - just train a model to compute the probability of data and reward using inverse of that
- locality of features
    - instead of having n possible discrete states
    - we now have n possible features that take on k discrete values
    - now we can learn the SR for each feature 
        - meaning what values are taken by this feature when transitioning from this starting state
        - (seems like we can use a convolutional network?? with down and upsampling)
    
    --> does this make sense?
    --> why would it be better than what we have before?

'''