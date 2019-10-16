import numpy as np
from keras.models import Sequential,Model
from keras.layers import Flatten,Input,Dense

EPOCH_PRETRAIN_POLICY=4
EPOCH_PRETRAIN_MODEL=5
EPOCH_FORMAL_TRAIN_POLICY=5
EPOCH_FORMAL_TRAIN_MODEL=4
EPISODES=10


class Policy():
    """
        The entire reinforcement learning model, including the agent and environment

        """
    def __init__(self,input_shape,classes,*args):
        """
        Initialization method
        :param Input_shape:iterable, the input of the model, the shape of all the input of the model must be the same
        :param Classes: int, number of categories (currently only supports ensemble of the classification model)
        :param *args: variable parameters, each input object must be compile keras model
            """
        assert type(classes) != 'int' or classes <= 1

        self.model_list=list(args)
        flatten_length = 1
        for i in input_shape:
            flatten_length *= i
        self.agent = Sequential()
        self.agent.add(Flatten(input_shape=input_shape))
        self.agent.add(Dense(round(flatten_length / 2), activation='relu'))
        self.agent.add(Dense(len(self.model_list), activation='softmax'))
        self.agent.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def train_policy(self,training_set,training_labels):
        """
            Reinforcement learning part, the agent do the action to divide the data set for each ensemble model
            :param training_set:training set, the shape must match the input_shape
            :param training_labels:training labels, must be one-hot encoding
            """
        # pre-train all classifier model
        for model in self.model_list:
            model.fit(training_set,training_labels,batch_size=100,epochs=EPOCH_PRETRAIN_MODEL,verbose=2)

        # compute reward according to the softmax output of each model on training set
        step_reward=[]
        distributed_training_set=[[]]*len(self.model_list)
        distributed_training_labels=[[]]*len(self.model_list)
        for model in  self.model_list:
            softmax_distribution=model.predict(training_set)
            print(softmax_distribution[:10,:])
            class_probability=np.max(softmax_distribution*training_labels,axis=1)
            print(class_probability.shape)
            step_reward.append(class_probability)
        step_reward=np.transpose(np.asarray(step_reward))

        print(step_reward[:10,:])

        # apply the agent, put the training set in to different parts for each model
        assert training_set.shape[0]==step_reward.shape[0]
        self.agent.fit(training_set, step_reward, epochs=EPOCH_PRETRAIN_POLICY, verbose=2)
        classes=self.agent.predict_classes(training_set)
        for i in range(training_set.shape[0]):
            distributed_training_set[classes[i]].append(training_set[i])
            distributed_training_labels[classes[i]].append(training_labels[i])
        for i in range(len(distributed_training_set)):
            distributed_training_set[i] = np.asarray(distributed_training_set[i])
            distributed_training_labels[i] = np.asarray(distributed_training_labels[i])

        # formal_train part
        for i in range(EPISODES):
            # train model with each training set
            for i,model in enumerate(self.model_list):
                if distributed_training_set[i].shape[0]>100:
                    model.fit(distributed_training_set[i],distributed_training_labels[i],epochs=EPOCH_FORMAL_TRAIN_MODEL,batch_size=100,verbose=2)

            # build reward for agent network
            step_reward = []
            for model in self.model_list:
                softmax_distribution = model.predict(training_set)
                class_probability = np.max(softmax_distribution * training_labels,axis=1)
                step_reward.append(class_probability)
            step_reward = np.transpose(np.asarray(step_reward))

            # train the agent and do actions on each elements in training set
            distributed_training_set = [[]] * len(self.model_list)
            distributed_training_labels = [[]] * len(self.model_list)
            self.agent.fit(training_set, step_reward, epochs=EPOCH_FORMAL_TRAIN_POLICY, batch_size=100, verbose=2)
            classes = self.agent.predict_classes(training_set)
            for i in range(training_set.shape[0]):
                distributed_training_set[classes[i]].append(training_set[i])
                distributed_training_labels[classes[i]].append(training_labels[i])
            for i in range(len(distributed_training_set)):
                distributed_training_set[i] = np.asarray(distributed_training_set[i])
                distributed_training_labels[i] = np.asarray(distributed_training_labels[i])

    def predict_with_policy(self,testing_set):
        """
            Use this ensemble model to predict on testing data set
            :param testing_set: testing data set, the shape must match input_shape
            :return: the output of model
            """
        action=self.agent.predict_classes(testing_set)
        return map(lambda index,instance:self.model_list[action[index]].predict(instance),enumerate(testing_set))

    def validate_with_policy(self,testing_set,testing_labels):
        """
                Use this ensemble model to validate the performance on testing data set
                    :param testing_set: testing data set, the shape must match input_shape
                    :param testing_labels: testing labels
                    :return:accuracy on testing data set
                    """
        predictions=self.predict_with_policy(testing_set)
        accuracy=np.argmax(predictions)
        testing_labels=np.argmax(testing_labels)
        return sum(accuracy==testing_labels)/testing_set.shape[0]

    def get_action(self,testing_set):
        return self.agent.predict_classes(testing_set)

    def get_agent(self):
        return self.agent

    def get_model_list(self):
        return  self.model_list

    def save(self,name_list):
        assert len(self.model_list)==len(name_list)
        for i,name in enumerate(name_list):
            self.model_list[i].save("model/"+name+"_model.h5")
        self.agent.save("model/agent.h5")
