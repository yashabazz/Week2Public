import pandas as pd
import numpy as np

def read_data_train():
  """ Read data """

  # Fixed params
  n_class = 6
  n_steps = 128

  label_path ='https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/y_train.txt'
  labels = pd.read_csv(label_path, header = None)

  list_of_channels = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x',
  'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z']

  X = np.zeros((len(labels), n_steps, len(list_of_channels)))

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_x_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,0] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_y_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,1] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_z_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,2] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_x_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,3] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_y_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,4] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_z_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,5] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_x_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,6] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_y_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,7] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_z_train.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,8] = dat_.to_numpy()


  # Return 
  return X, labels[0].values, list_of_channels

def read_data_test():
  """ Read data """

  # Fixed params
  n_class = 6
  n_steps = 128

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/y_test.txt'
  labels = pd.read_csv(label_path, header = None)

  list_of_channels = ['body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x',
  'body_gyro_y', 'body_gyro_z', 'total_acc_x', 'total_acc_y', 'total_acc_z']

  X = np.zeros((len(labels), n_steps, len(list_of_channels)))

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_x_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,0] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_y_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,1] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_acc_z_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,2] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_x_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,3] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_y_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,4] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/body_gyro_z_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,5] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_x_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,6] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_y_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,7] = dat_.to_numpy()

  label_path = 'https://raw.githubusercontent.com/BeaverWorksMedlytics2020/Data_Public/master/NotebookExampleData/Week2/week2_conv1d/IntertialSignals/total_acc_z_test.txt'
  dat_ = pd.read_csv(label_path, delim_whitespace = True, header = None)
  X[:,:,8] = dat_.to_numpy()


  # Return 
  return X, labels[0].values, list_of_channels