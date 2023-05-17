# Deep Understanding Of Deep Learning

Python code accompanying the course "A deep understanding of deep learning (with Python intro)"

Master deep learning in PyTorch using an experimental scientific approach, with lots of examples and practice problems.

See https://www.udemy.com/course/deeplearning_x/?couponCode=202302 for more details, preview videos, and to enroll in the full course.

- [Deep Understanding Of Deep Learning](#deep-understanding-of-deep-learning)
  - [Math](#math)
  - [Gradient Descent](#gradient-descent)
  - [ANN](#ann)
  - [Overfitting and cross-validation](#overfitting-and-cross-validation)
  - [Regularization](#regularization)
  - [Meta Parameters (activations,optimizers)](#meta-parameters-activationsoptimizers)
  - [FFN (Feed-Forward-Networks)](#ffn-feed-forward-networks)
  - [More On Data](#more-on-data)
  - [Measuring Model Performance](#measuring-model-performance)
  - [FFN Milestone Projects](#ffn-milestone-projects)
  - [Weight Inits And Investigation](#weight-inits-and-investigation)
  - [Autoencoders](#autoencoders)
  - [Running models on a GPU](#running-models-on-a-gpu)
  - [Convolution And Transformation](#convolution-and-transformation)
  - [Understanding CNN And Design CNNs](#understanding-cnn-and-design-cnns)
  - [CNN Milestone Projects](#cnn-milestone-projects)
  - [Transfer Learning](#transfer-learning)
  - [Style Transfer](#style-transfer)
  - [GANs](#gans)
  - [RNNs](#rnns)

## Math

- [UDL_math_transpose.ipynb](./math/DUDL_math_transpose.ipynb)
- [UDL_math_dotproduct.ipynb](./math/DUDL_math_dotproduct.ipynb)
- [UDL_math_matrixMult.ipynb](./math/DUDL_math_matrixMult.ipynb)
- [UDL_math_softmax.ipynb](./math/DUDL_math_softmax.ipynb)
- [UDL_math_log.ipynb](./math/DUDL_math_log.ipynb)
- [UDL_math_entropy.ipynb](./math/DUDL_math_entropy.ipynb)
- [UDL_math_argmin.ipynb](./math/DUDL_math_argmin.ipynb)
- [UDL_math_meanvar.ipynb](./math/DUDL_math_meanvar.ipynb)
- [UDL_math_sampling.ipynb](./math/DUDL_math_sampling.ipynb)
- [UDL_math_randomseed.ipynb](./math/DUDL_math_randomseed.ipynb)
- [UDL_math_ttest.ipynb](./math/DUDL_math_ttest.ipynb)
- [UDL_math_derivatives1.ipynb](./math/DUDL_math_derivatives1.ipynb)
- [UDL_math_derivatives2.ipynb](./math/DUDL_math_derivatives2.ipynb)

## Gradient Descent

- [UDL_GradientDescent_1D.ipynb](./gradientDescent/DUDL_GradientDescent_1D.ipynb)
- [UDL_GradientDescent_CodeChallengeStartValue.ipynb](./gradientDescent/DUDL_GradientDescent_CodeChallengeStartValue.ipynb)
- [UDL_GradientDescent_2D.ipynb](./gradientDescent/DUDL_GradientDescent_2D.ipynb)
- [UDL_GradientDescent_experiment.ipynb](./gradientDescent/DUDL_GradientDescent_experiment.ipynb)
- [UDL_GradientDescent_codeChallenge_lr.ipynb](./gradientDescent/DUDL_GradientDescent_codeChallenge_lr.ipynb)

## ANN

- [UDL_ANN_regression.ipynb](./ANN/DUDL_ANN_regression.ipynb)
- [UDL_ANN_codeChallenge_regression.ipynb](./ANN/DUDL_ANN_codeChallenge_regression.ipynb)
- [UDL_ANN_classifyQwerties.ipynb](./ANN/DUDL_ANN_classifyQwerties.ipynb)
- [UDL_ANN_learningrates.ipynb](./ANN/DUDL_ANN_learningrates.ipynb)
- [UDL_ANN_multilayer.ipynb](./ANN/DUDL_ANN_multilayer.ipynb)
- [UDL_ANN_multioutput.ipynb](./ANN/DUDL_ANN_multioutput.ipynb)
- [UDL_ANN_codeChallengeQwerties.ipynb](./ANN/DUDL_ANN_codeChallengeQwerties.ipynb)
- [UDL_ANN_nHiddenUnits.ipynb](./ANN/DUDL_ANN_nHiddenUnits.ipynb)
- [UDL_ANN_numParameters.ipynb](./ANN/DUDL_ANN_numParameters.ipynb)
- [UDL_ANN_breadthVsDepth.ipynb](./ANN/DUDL_ANN_breadthVsDepth.ipynb)
- [UDL_ANN_seqVsClass.ipynb](./ANN/DUDL_ANN_seqVsClass.ipynb)
- [UDL_ANN_codeChallengeSeq2class.ipynb](./ANN/DUDL_ANN_codeChallengeSeq2class.ipynb)

## Overfitting and cross-validation

- [UDL_overfitting_manual.ipynb](./overfitting/DUDL_overfitting_manual.ipynb)
- [UDL_overfitting_scikitlearn.ipynb](./overfitting/DUDL_overfitting_scikitlearn.ipynb)
- [UDL_overfitting_dataLoader.ipynb](./overfitting/DUDL_overfitting_dataLoader.ipynb)
- [UDL_overfitting_trainDevsetTest.ipynb](./overfitting/DUDL_overfitting_trainDevsetTest.ipynb)
- [UDL_overfitting_regression.ipynb](./overfitting/DUDL_overfitting_regression.ipynb)

## Regularization

- [UDL_regular_dropout.ipynb](./regularization/DUDL_regular_dropout.ipynb)
- [UDL_regular_dropoutInPytorch.ipynb](./regularization/DUDL_regular_dropoutInPytorch.ipynb)
- [UDL_regular_dropout_example2.ipynb](./regularization/DUDL_regular_dropout_example2.ipynb)
- [UDL_regular_L1regu.ipynb](./regularization/DUDL_regular_L1regu.ipynb)
- [UDL_regular_L2regu.ipynb](./regularization/DUDL_regular_L2regu.ipynb)
- [UDL_regular_minibatch.ipynb](./regularization/DUDL_regular_minibatch.ipynb)
- [UDL_regular_testBatchT2.ipynb](./regularization/DUDL_regular_testBatchT2.ipynb)
- [UDL_regular_codeChallenge_minibatch.ipynb](./regularization/DUDL_regular_codeChallenge_minibatch.ipynb)

## Meta Parameters (activations,optimizers)

- [UDL_metaparams_intro2winedata.ipynb](./metaparams/DUDL_metaparams_intro2winedata.ipynb)
- [UDL_metaparams_codeChallengeDropout.ipynb](./metaparams/DUDL_metaparams_codeChallengeDropout.ipynb)
- [UDL_metaparams_batchNorm.ipynb](./metaparams/DUDL_metaparams_batchNorm.ipynb)
- [UDL_metaparams_CodeChallengeBatches.ipynb](./metaparams/DUDL_metaparams_CodeChallengeBatches.ipynb)
- [UDL_metaparams_ActivationFuns.ipynb](./metaparams/DUDL_metaparams_ActivationFuns.ipynb)
- [UDL_metaparams_ActivationComparisons.ipynb](./metaparams/DUDL_metaparams_ActivationComparisons.ipynb)
- [UDL_metaparams_CodeChallengeRelus.ipynb](./metaparams/DUDL_metaparams_CodeChallengeRelus.ipynb)
- [UDL_metaparams_CodeChallenge_sugar.ipynb](./metaparams/DUDL_metaparams_CodeChallenge_sugar.ipynb)
- [UDL_metaparams_loss.ipynb](./metaparams/DUDL_metaparams_loss.ipynb)
- [UDL_metaparams_multioutput.ipynb](./metaparams/DUDL_metaparams_multioutput.ipynb)
- [UDL_metaparams_momentum.ipynb](./metaparams/DUDL_metaparams_momentum.ipynb)
- [UDL_metaparams_optimizersComparison.ipynb](./metaparams/DUDL_metaparams_optimizersComparison.ipynb)
- [UDL_metaparams_CodeChallengeOptimizers.ipynb](./metaparams/DUDL_metaparams_CodeChallengeOptimizers.ipynb)
- [UDL_metaparams_CodeChallengeAdamL2.ipynb](./metaparams/DUDL_metaparams_CodeChallengeAdamL2.ipynb)
- [UDL_metaparams_learningRateDecay.ipynb](./metaparams/DUDL_metaparams_learningRateDecay.ipynb)

## FFN (Feed-Forward-Networks)

- [UDL_FFN_aboutMNIST.ipynb](./FFN/DUDL_FFN_aboutMNIST.ipynb)
- [UDL_FFN_FFNonMNIST.ipynb](./FFN/DUDL_FFN_FFNonMNIST.ipynb)
- [UDL_FFN_CodeChallenge_binMNIST.ipynb](./FFN/DUDL_FFN_CodeChallenge_binMNIST.ipynb)
- [UDL_FFN_CodeChallenge_normalization.ipynb](./FFN/DUDL_FFN_CodeChallenge_normalization.ipynb)
- [UDL_FFN_weightHistograms.ipynb](./FFN/DUDL_FFN_weightHistograms.ipynb)
- [UDL_FFN_CodeChallengeBreadthDepth.ipynb](./FFN/DUDL_FFN_CodeChallengeBreadthDepth.ipynb)
- [UDL_FFN_CodeChallenge_optimizers.ipynb](./FFN/DUDL_FFN_CodeChallenge_optimizers.ipynb)
- [UDL_FFN_scrambledMNIST.ipynb](./FFN/DUDL_FFN_scrambledMNIST.ipynb)
- [UDL_FFN_shiftedMNIST.ipynb](./FFN/DUDL_FFN_shiftedMNIST.ipynb)
- [UDL_FFN_CodeChallenge_missing7.ipynb](./FFN/DUDL_FFN_CodeChallenge_missing7.ipynb)

## More On Data

- [UDL_data_datasetLoader.ipynb](./data/DUDL_data_datasetLoader.ipynb)
- [UDL_data_dataVsDepth.ipynb](./data/DUDL_data_dataVsDepth.ipynb)
- [UDL_data_CodeChallengeUnbalanced.ipynb](./data/DUDL_data_CodeChallengeUnbalanced.ipynb)
- [UDL_data_oversampling.ipynb](./data/DUDL_data_oversampling.ipynb)
- [UDL_data_noiseAugmentation.ipynb](./data/DUDL_data_noiseAugmentation.ipynb)
- [UDL_data_featureAugmentation.ipynb](./data/DUDL_data_featureAugmentation.ipynb)
- [UDL_data_data2colab.ipynb](./data/DUDL_data_data2colab.ipynb)
- [UDL_data_saveLoadModels.ipynb](./data/DUDL_data_saveLoadModels.ipynb)
- [UDL_data_saveTheBest.ipynb](./data/DUDL_data_saveTheBest.ipynb)

## Measuring Model Performance

- [UDL_measurePerformance_APRF.ipynb](./measurePerformance/DUDL_measurePerformance_APRF.ipynb)
- [UDL_measurePerformance_APRFexample1.ipynb](./measurePerformance/DUDL_measurePerformance_APRFexample1.ipynb)
- [UDL_measurePerformance_example2.ipynb](./measurePerformance/DUDL_measurePerformance_example2.ipynb)
- [UDL_measurePerformance_codeChallenge_unequal.ipynb](./measurePerformance/DUDL_measurePerformance_codeChallenge_unequal.ipynb)
- [UDL_measurePerformance_time.ipynb](./measurePerformance/DUDL_measurePerformance_time.ipynb)

## FFN Milestone Projects

- [UDL_FFNmilestone_project1.ipynb](./FFNmilestone/DUDL_FFNmilestone_project1.ipynb)
- [UDL_FFNmilestone_project2.ipynb](./FFNmilestone/DUDL_FFNmilestone_project2.ipynb)
- [UDL_FFNmilestone_project3.ipynb](./FFNmilestone/DUDL_FFNmilestone_project3.ipynb)

## Weight Inits And Investigation

- [UDL_weights_matrixsizes.ipynb](./weights/DUDL_weights_matrixsizes.ipynb)
- [UDL_weights_demoinits.ipynb](./weights/DUDL_weights_demoinits.ipynb)
- [UDL_weights_codeChallenge_weightstd.ipynb](./weights/DUDL_weights_codeChallenge_weightstd.ipynb)
- [UDL_weights_XavierKaiming.ipynb](./weights/DUDL_weights_XavierKaiming.ipynb)
- [UDL_weights_CodeChallenge_XavierKaiming.ipynb](./weights/DUDL_weights_CodeChallenge_XavierKaiming.ipynb)
- [UDL_weights_codeChallenge_identicalRandom.ipynb](./weights/DUDL_weights_codeChallenge_identicalRandom.ipynb)
- [UDL_weights_freezeWeights.ipynb](./weights/DUDL_weights_freezeWeights.ipynb)
- [UDL_weights_weightchanges.ipynb](./weights/DUDL_weights_weightchanges.ipynb)

## Autoencoders

- [UDL_autoenc_denoisingMNIST.ipynb](./autoencoders/DUDL_autoenc_denoisingMNIST.ipynb)
- [UDL_autoenc_codeChallenge_Nunits.ipynb](./autoencoders/DUDL_autoenc_codeChallenge_Nunits.ipynb)
- [UDL_autoenc_occlusion.ipynb](./autoencoders/DUDL_autoenc_occlusion.ipynb)
- [UDL_autoenc_MNISTlatentCode.ipynb](./autoencoders/DUDL_autoenc_MNISTlatentCode.ipynb)
- [UDL_autoenc_tiedWeights.ipynb](./autoencoders/DUDL_autoenc_tiedWeights.ipynb)

## Running models on a GPU

- [UDL_GPU_implement.ipynb](./GPU/DUDL_GPU_implement.ipynb)
- [UDL_GPU_CodeChallenge2GPU.ipynb](./GPU/DUDL_GPU_CodeChallenge2GPU.ipynb)

## Convolution And Transformation

- [UDL_convolution_convInCode.ipynb](./convolution/DUDL_convolution_convInCode.ipynb)
- [UDL_convolution_conv2.ipynb](./convolution/DUDL_convolution_conv2.ipynb)
- [UDL_convolution_codeChallenge.ipynb](./convolution/DUDL_convolution_codeChallenge.ipynb)
- [UDL_convolution_conv2transpose.ipynb](./convolution/DUDL_convolution_conv2transpose.ipynb)
- [UDL_convolution_meanMaxPool.ipynb](./convolution/DUDL_convolution_meanMaxPool.ipynb)
- [UDL_convolution_transforms.ipynb](./convolution/DUDL_convolution_transforms.ipynb)
- [UDL_convolution_customDataSet.ipynb](./convolution/DUDL_convolution_customDataSet.ipynb)

## Understanding CNN And Design CNNs

- [UDL_CNN_CNN4MNIST.ipynb](./CNN/DUDL_CNN_CNN4MNIST.ipynb)
- [UDL_CNN_shiftedMNIST.ipynb](./CNN/DUDL_CNN_shiftedMNIST.ipynb)
- [UDL_CNN_GaussClass.ipynb](./CNN/DUDL_CNN_GaussClass.ipynb)
- [UDL_CNN_GaussClassFeatureMaps.ipynb](./CNN/DUDL_CNN_GaussClassFeatureMaps.ipynb)
- [UDL_CNN_codeChallengeSoftcoding.ipynb](./CNN/DUDL_CNN_codeChallengeSoftcoding.ipynb)
- [UDL_CNN_CodeChallengeLinearUnits.ipynb](./CNN/DUDL_CNN_CodeChallengeLinearUnits.ipynb)
- [UDL_CNN_GaussAE.ipynb](./CNN/DUDL_CNN_GaussAE.ipynb)
- [UDL_CNN_CodeChallengeAEocclusion.ipynb](./CNN/DUDL_CNN_CodeChallengeAEocclusion.ipynb)
- [UDL_CNN_codeChallengeCustomLoss.ipynb](./CNN/DUDL_CNN_codeChallengeCustomLoss.ipynb)
- [UDL_CNN_findGauss.ipynb](./CNN/DUDL_CNN_findGauss.ipynb)
- [UDL_CNN_EMNIST.ipynb](./CNN/DUDL_CNN_EMNIST.ipynb)
- [UDL_CNN_codeChallengeBeatThis.ipynb](./CNN/DUDL_CNN_codeChallengeBeatThis.ipynb)
- [UDL_CNN_codeChallengeNumChans.ipynb](./CNN/DUDL_CNN_codeChallengeNumChans.ipynb)

## CNN Milestone Projects

- [UDL_CNNmilestone_project1.ipynb](./CNNmilestone/DUDL_CNNmilestone_project1.ipynb)
- [UDL_CNNmilestone_project2.ipynb](./CNNmilestone/DUDL_CNNmilestone_project2.ipynb)
- [UDL_CNNmilestone_project3.ipynb](./CNNmilestone/DUDL_CNNmilestone_project3.ipynb)
- [UDL_CNNmilestone_project4.ipynb](./CNNmilestone/DUDL_CNNmilestone_project4.ipynb)

## Transfer Learning

- [UDL_transfer_MNISTtoFMNIST.ipynb](./transferlearning/DUDL_transfer_MNISTtoFMNIST.ipynb)
- [UDL_transfer_codeChallenge_letters2numbers.ipynb](./transferlearning/DUDL_transfer_codeChallenge_letters2numbers.ipynb)
- [UDL_transfer_resnet.ipynb](./transferlearning/DUDL_transfer_resnet.ipynb)
- [UDL_transfer_codeChallengeVGG16.ipynb](./transferlearning/DUDL_transfer_codeChallengeVGG16.ipynb)
- [UDL_transfer_pretrainFMNIST.ipynb](./transferlearning/DUDL_transfer_pretrainFMNIST.ipynb)
- [UDL_transfer_PretrainCIFAR.ipynb](./transferlearning/DUDL_transfer_PretrainCIFAR.ipynb)

## Style Transfer

- [UDL_style_screamingBathtub.ipynb](./styletransfer/DUDL_style_screamingBathtub.ipynb)
- [UDL_style_codeChallengeAlexNet.ipynb](./styletransfer/DUDL_style_codeChallengeAlexNet.ipynb)

## GANs

- [UDL_GAN_MNIST.ipynb](./GANs/DUDL_GAN_MNIST.ipynb)
- [UDL_GAN_CNNganGaus.ipynb](./GANs/DUDL_GAN_CNNganGaus.ipynb)
- [UDL_GAN_codeChallengeGaus.ipynb](./GANs/DUDL_GAN_codeChallengeGaus.ipynb)
- [UDL_GAN_CNNganFMNIST.ipynb](./GANs/DUDL_GAN_CNNganFMNIST.ipynb)
- [UDL_GAN_codeChallengeFMNIST.ipynb](./GANs/DUDL_GAN_codeChallengeFMNIST.ipynb)
- [UDL_GAN_codeChallengeCIFAR.ipynb](./GANs/DUDL_GAN_codeChallengeCIFAR.ipynb)
- [UDL_GAN_codeChallengeFaces.ipynb](./GANs/DUDL_GAN_codeChallengeFaces.ipynb)

## RNNs

- [UDL_RNN_intro2RNN.ipynb](./RNN/DUDL_RNN_intro2RNN.ipynb)
- [UDL_RNN_altSequences.ipynb](./RNN/DUDL_RNN_altSequences.ipynb)
- [UDL_RNN_codeChallenge_SineExtrapolate.ipynb](./RNN/DUDL_RNN_codeChallenge_SineExtrapolate.ipynb)
- [UDL_RNN_LSTMGRU.ipynb](./RNN/DUDL_RNN_LSTMGRU.ipynb)
- [UDL_RNN_loremipsum.ipynb](./RNN/DUDL_RNN_loremipsum.ipynb)
