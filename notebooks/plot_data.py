import pandas as pd
import matplotlib.pyplot as plt
import os

class PlotActivity():
    '''PlotActivity plots a experiment with the entered activity'''

    def __init__(self,experiment):

        self.experiment = str(experiment)

        if int(self.experiment) < 10:
            self.experiment = '0' + self.experiment


        for i in os.listdir('../data/RawData/'):
            if "acc_exp" + self.experiment in i:
                path = "../data/RawData/" + i

        self.df = pd.read_csv(path, sep=" ", names=['x','y','z'])
        self.df["time_s"] = self.df.index/50
        self.df = self.df[['time_s','x','y','z']]

        self.labels = pd.read_csv('../data/RawData/labels.txt', sep=" ", header=None)
        self.labels.columns = ['experiment','person','activity','start','end']

    def read_activity_labels(self):
        '''Read the activity labels.'''

        labels_dict =  open('../data/activity_labels.txt','r')
        labels_dict = labels_dict.read().split('\n')

        my_dict = {}

        for i in labels_dict[:-1]:
            key = i.split()[0]
            value = i.split()[1]
            my_dict[key] = value

        return my_dict



    def plot_raw_data(self, start, end, axis = ['x','y','z'],  experiment = None, person = None,activity=None):
        '''Plot the raw sensor data.

        args:

        df: the dataframe with the sensor data. The columns 'x','y','z' are required.
        start: the number of the sample as a start point, default 0
        end: the number of the sample as a end point, default 1000
        axis: the axis to plot

        return:
        Returns a matplotlib diagramm
        '''

        df = self.df
        labels_dict = self.read_activity_labels()

        fig,ax = plt.subplots(len(axis),1)
        colors = ['red','blue','green']

        for i in range(len(axis)):

            ax[i].plot(df.index[start:end], df[axis[0]][start:end], ls = '-',c=colors[i])
            ax[i].set_xlabel('sample')
            ax[i].set_ylabel('acceleration [$g/ms^2$]')
            ax[i].set_title("experiment: {} person: {}".format(experiment,person))

            if activity != None:

                ax[i].legend(['Nr.: ' + activity + ' ' + labels_dict[activity]])

            ax[i].grid()

        plt.tight_layout()


    def plot_activity(self, activity = 1):
        '''Plots the one activity from a experiment

        args:

        experiment: the id of the experiment
        activity: the id of the activity

        returns:
        plot of the 3 axes of the sensor

        '''

        label_to_plot = self.labels.loc[(self.labels['experiment'] == int(self.experiment)) & (self.labels['activity'] == activity)]
        for index, row  in label_to_plot.iterrows():

            plt.figure()

            self.plot_raw_data(
                          start=row['start'],
                          end=row['end'],
                          experiment=row['experiment'],
                          person=row['person'],
                          activity=str(row['activity'])
                         )

            plt.show()
