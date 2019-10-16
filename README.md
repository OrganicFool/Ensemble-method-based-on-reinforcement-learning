# Written in front:
之前参加kaggle的时候（其实是水。。）想到的一个ensemble策略。因为当时正好在做强化学习去噪，就觉得这个场景其实和ensemble有点像。具体的想法就是通过强化学习的方法，让agent将每一个样本（state）分给不同的学习器（action），再通过在不同学习器上的表现定义reward，直到每个样本都尽可能被分给在他上面表现最好的学习器。用比较intutive的方式来解释的话，就是即使在同一个任务上（比如文本分类），不同的人擅长的方面也不同（rnn结构可能擅长处理序列，cnn结构可能擅长提取特征（好吧其实我也不知道cnn擅长啥-_-））,如果能物尽其用，让每个样本被最擅长他的分类器处理，就能提高在整个数据集上的表现。

I thought of this ensemble strategy, When I participated in kaggle. Because I was doing reinforcement learning to denoise at the time, I felt that this scene is actually a little bit like ensemble. The specific idea is to design a reinforcement learning process, let the agent assign each sample(state) to different classifier(action), and then define reward through the performance on different learners, until each sample is given as much as possible. He showed the best learner on it. Explain in a way that is more intutive, that is, even on the same task (such as text classification), different people are good at different aspects (rnn structure may be good at processing sequences, cnn structure may be good at extracting features (well, actually I also don't know what is cnn good at-_-)). If I can make the best use of it, let each sample be best handled by his classifier, it can improve the performance on the whole data set, so I designed an agent to be different. The sample is assigned to the classifier that is best at him.
## Description
To demonstrate, I chose the very simple data set cifar-10 and three very common neural networks: two-layer CNN, full-connected neural network and VGG16. You can build very large classification tasks or choose very complex classification models and even modify the network structure of the policy part. The hyperparameters are set at the top of each script.
## Version
keras:2.2.4
tensorflow:1.13.1(it only ran on gpu before)
## Usage
**Step 1.
Install [Keras 2.2.4](https://github.com/fchollet/keras) 
with [TensorFlow](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu
pip install keras==2.2.4
```

**Step 2. Clone this repository to local.**
```
git clone https://github.com/AntiBupt/Ensemble-method-based-on-reinforcement-learning.git
cd Ensemble-method-based-on-reinforcement-learning
```
**Step 3. Train the model with default configuration(cifar10+cnn+dense+vgg16).**
```
python train.py
```
## Current problems(10/16/2019)
1.a lot of hyperparameters can't be Adjusted

2.Validation script

3.it only supportted classification and numpy input

4.No available checkpoint
## Contact me
```
E-mail:xiahandong6250@bupt.edu.cn
WeChat:xhd19990625
```
