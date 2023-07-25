# A deep understanding of deep learning

- [A deep understanding of deep learning](#a-deep-understanding-of-deep-learning)
  - [Measuring model performance](#measuring-model-performance)
    - [Two perspective of the world](#two-perspective-of-the-world)
      - [Experiment: Did you see a cat?](#experiment-did-you-see-a-cat)
      - [Alternate terms from signal detection](#alternate-terms-from-signal-detection)
      - [The problem with accuracy](#the-problem-with-accuracy)

## Measuring model performance

### Two perspective of the world

> - The main idea behind signal-detection theory
> - The four categories of responses in signal-dtection theory.
> - Why signal-detection theory is useful for measuring performance DL

So far in this course we have been evaluating the model performance using two matrix, losses and accuracy.

We haven't really interpreted them that much except say that losses should go down and accuracy should go out.

So it turns out there are additional measures of model performance they give us increasing insight more sensitivity of how exacatly the model is performing.

That is the major goal of this section of this course.

and in this video I'm going to give you a little bit of background about signal detection theory.

signal detection theory is widely used in 심리학, economics, and machine learning to understand performance of predictive model and categorical decisions.

One of the fundamental backbone, really the skeleton of signal detection theory is a matrix as called four categories of responses which comes out another concept called the two perspectives of the world.

So in this video I'm going to introduce you to these concept which will give you firm grounding in signal detection theory 

and that is going to allow you to understand the different measures of model performance that you'll learn about later on in this section.

#### Experiment: Did you see a cat?

![](.md/README4.md/2023-07-25-22-59-24.png)

So let's begin here we have our personify neural network this is our deep learning neural network 

and we're asking our neural network to look at pictures and basically just say if that picture contains a cat or if it doesn't contain a cat.

So here is a picture of a cat, here is a picture of a boat which is definitely not the same thing as as cat.

So the model looks at the picture and she just says yes, it's a cat or no it's not a cat.

so we have two perspectives of the world.

we have the reality in the columns 

and that is the picture actually is a cat or boat

and then we have the model's output which is the prediction about the state of the world.

that is either a cat or a boat.

so when the model says it's a cat, and the reality is that actually was a cat.

so the model looks this picture and says, 'yes, this is a cat.'

then that is a correct answer, we call that correct answer a Hit.

there's alternative terminology that I'll introduce you to in a moment but

it's commonly called a hit.

Now this is not the only way of giving a correct answer

it's also possible fo, you know, the actual picture to be a boat 

and the model says, that was a boat.

that is called a coprrect rejection over here.

so this is not a cat and the model says it is not a cat so a correct rejection and hits these are two different ways of being correct.

then of course we have two different ways of the model being incorrect, making a mistake 

first we have was called a Miss and that is where the picture actually was a cat, the reality was a cat but the model looked this picture and she said,  you know, i think that looks like a boat.

I'm gonna say a boat.

so she was wrong and we call this category a Miss.

and then of course, this thing over here, this is called a False alarm.

This is where the reality was the boat, we show the model the picture of a boat and she said, you know, i'm gonna go with a cat.

that looks to me like a cat.

Maybe she's just trained yet.

So this is another way of being incorrect, we call this a false alram.

so all of this here, this is one set of terminology that is used in signal detection theory.

It turns out the terminology varies a little bit in different literatures.

#### Alternate terms from signal detection

![](.md/README4.md/2023-07-25-23-17-38.png)

So now i'm gonna show you different terms for exactly the same concept.

Concepts are not different, just the wording is slightly different.

so sometimes we call this objective reality, you know, interesting philosophical<sup>철학적인</sup> discussion with more on ethical implications of whether objective reality exists what that means

I think you know what i mean here.

Object reality is present or absent in this case it will be a cat

the cat was present or the cat was absent

and then we have the subjective reality which is the prediction or output of the model ant that is yes, it was a cat, or no, it wasn't a cat.

Here we have the other terms are a little different 

this is, i'm going back to the previous slide.

So here we call hits and correct rejections

here these are called true positive and true negative, 

here's false alarms and misses

and they're also called false positives and false negatives.

Again I apologize for the confusion of the different terminology but that is 

just unfortunate state of a ... in human civilization.

ok now this entire matrix, once we put numbers into this, you'll see examples of, numerical examples starting in the next video.

We can call this thing a confusion matrix.

it's called a confusion matix because it shows all the ways of confusing the subjective reality, the model prediction with the reality objective which is the target variables.

So more on confusion matrix is later on in this section.

Now why do we need to worry about these 4 different categories of responses.

#### The problem with accuracy

We already know how to compute accuracy, so why isn't that enough?

Well, what part of the answer comes from a video that you saw in the previous section about unbalanced design.

I remember here, I said, that the model can smply say a cat all the time for every picture and it will be correct 99% of the time.

This is not wrong, the true accruacy really is 99 %.

but that's not an informative number in this case because of this unbalanced design.

so the conclusion here is that accuracy is very useful, accuracy is a very informative measure but it also hides possible biases in the model performance and also due to the nature of the design, if the data set is unbalanced.

So therefore what we want is to have additional measures 

we want more tools and tool box to be able to supplement a measure of accuracy with other measures of performance that can reveal possible biases or sensitivy in the data.
