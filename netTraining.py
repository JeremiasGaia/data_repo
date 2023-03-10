# Load the Drive helper and mount
import pandas as pd
import numpy as np
import os
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from tensorflow import keras
import mypreprocessing as pp
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# import visualkeras




# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
 	X, y = list(), list()
 	for i in range(len(sequences)):
         end_ix = i + n_steps
         
         if end_ix > len(sequences):
             break 
         seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -3:]
         X.append(seq_x)
         y.append(seq_y)
 	return array(X), array(y)


def training(train_data_path, valid_data_path, test_data_path, run_number, run_log_path, training_epochs):
    
    i=run_number
    training_file   = os.path.join(train_data_path, 'train_data_run_'+str(i)+'.xlsx')
    validation_file = os.path.join(valid_data_path, 'validation_data_run_'+str(i)+'.xlsx')
    testing_file    = os.path.join(test_data_path , 'test_data_run_'+str(i)+'.xlsx')
   
    model_save_filename  = os.path.join(run_log_path,'run_models' , 'modelStruct_run_' + str(i) + '.json')
    weights_save_filename= os.path.join(run_log_path,'run_models' , 'modelWeights_run_' + str(i) + '.h5')
    
    cm_save_filename = os.path.join(run_log_path,'run_results', 'confusion_matrices', 'CM_run_' + str(i) + '.pdf') #confusion matrix
    tr_hist_filename = os.path.join(run_log_path,'run_results', 'train_history', 'training_history_run_' + str(i) + '.xlsx') #file to store training information 
    
    
    
    df_train = pd.read_excel(training_file, index_col=0)
    df_test  = pd.read_excel(testing_file , index_col=0)
    df_valid = pd.read_excel(validation_file , index_col=0)
    
    train_dataset = df_train.to_numpy()
    valid_dataset = df_valid.to_numpy()
    test_dataset  = df_test.to_numpy()
    
    # aplico Z-score a cada columna
    [proc_train_data, proc_test_data, proc_valid_data] = pp.zscore(train_dataset[:,:-1], test_dataset[:,:-1], valid_dataset[:,:-1]) #todas las columnas menos la ultima
    
    #la tangente hiperbólica ayuda a mantener los valores entre 1 y -1
    train_data = np.tanh(proc_train_data)
    test_data  = np.tanh(proc_test_data)
    valid_data = np.tanh(proc_valid_data)
    
    tra_labels = pp.get_labeled_data(train_dataset[:,-1])
    tes_labels = pp.get_labeled_data(test_dataset[:,-1])
    val_labels = pp.get_labeled_data(valid_dataset[:,-1])
    
    
    train_data = np.hstack([train_data, tra_labels ]) 
    test_data  = np.hstack([test_data , tes_labels ])
    valid_data = np.hstack([valid_data, val_labels ])
    
    # #choose a number of time steps
    n_steps = 3
    # # convert into input/output
    
    X_train, y_train = split_sequences(train_data, n_steps)
    X_valid, y_valid = split_sequences(valid_data, n_steps)
    X_test, y_test   = split_sequences(test_data , n_steps)
    # The first dimension of X is the number of samples, same as the input dataset length. 
    # The second dimension of X is the number of time steps per sample, in this case = n_steps, the value specified to the function. 
    # Finally, the last dimension of X specifies the number of parallel time series or variables (columns) , same as the input dataset column number.
    
    
    n_features = X_train.shape[2]
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    #  MODEL DEFINITION
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    lstm_units  = 20
    dense_units = 10 #orig 256
    drop_perc= 0.4 #dropout percentage
    
    # probar con sigmoid, elu, relu
    
    model = Sequential()
    model.add(LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(drop_perc))
    model.add(LSTM(lstm_units, activation='tanh'))
    
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(drop_perc))
    model.add(Dense(3, activation='softmax'))
    #probar con salida lineal, que es una combinación lineal de las características
    
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # opt = keras.optimizers.Adam()
    # model.compile(optimizer=opt, loss='mse')
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    model.summary()
    
    # fit model
    history= model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=training_epochs, verbose=1)

    

    
    #%%
    
    """
    EVALUACIÓN DE DESEMPEÑO CON CONFUSION MATRIX
    
    """
    predictions_raw= model.predict(X_test);#probamos predecir con los datos de test
    
    predictions=np.argmax(predictions_raw, axis=1)  #transformo las predicciones en enteros
    
    y_pred = predictions; 
    
       #ACA COLOCAR LOS DATOS REALES DE SALIDA DE LOS VALORES CON LOS QUE PREDECIMOS
    
    y_true = np.argmax(y_test, axis=1); #estas son las salidas reales(transformadas en enteros)
    
    lab=[0,1,2];
    # Calculate the confusion matrix.
    CM=confusion_matrix(y_true, y_pred,labels=lab);
      
    
    # ------------------------------------------------------------------------------
    #                            ANÁLISIS DE PREDICCIÓN
    # ------------------------------------------------------------------------------
    
    # lab=['Normal','M_hole','DR','CSR'];
    lab=['Lost','Work','Other']
    class_names=lab
    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=class_names)
    disp.plot()
    plt.savefig(fname = cm_save_filename)

    
    
    # # plt.savefig(fname='Matriz_de_Confusion_7030_CNN.pdf')
    # # plt.savefig(fname='pngMatriz_de_Confusion_7030_CNN.png')
    # ann_viz(model, title="Network")
    # visualkeras.layered_view(model, legend=True)
    
    #%% SAVING DATA
    # serialize model to JSON
    model_json = model.to_json()

                 
    with open(model_save_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_save_filename)
    print("The model has been saved ...")
    
    plt.close('all')
    
#  ********************************************************************************
    info_df = pd.DataFrame({
        'training_loss':        history.history['loss'],
        'training_acc':         history.history['accuracy'],
        'validation_loss':      history.history['val_loss'],
        'validation_acc':       history.history['val_accuracy'] })
        
    writer = pd.ExcelWriter(tr_hist_filename, engine='xlsxwriter')# Create a Pandas Excel writer

    # Write the dataframe data to XlsxWriter. Turn off the default header and
    # index and skip one row to allow us to insert a user defined header.
    info_df.to_excel(writer, sheet_name='Sheet1', startrow=1, header=False, index=False)


    workbook = writer.book     # Get the xlsxwriter workbook and worksheet objects.
    worksheet = writer.sheets['Sheet1']


    (max_row, max_col) = info_df.shape   # Get the dimensions of the dataframe.

    column_settings = [{'header': column} for column in info_df.columns]# Create a list of column headers

    worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})# Add the Excel table structure.

    worksheet.set_column(0, max_col - 1, 12)# Make the columns wider.

    writer.save()# Close the Pandas Excel writer and output the Excel file.
#  ********************************************************************************
    
    
    
    return


















