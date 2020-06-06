from func_data_preparation import *
from func_ml import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--save_as", action = 'store', default = "../models/har_model_v10.pkl" ,
                    help='Give the name of the trained model. Default: ../models/har_model_v10.pkl')
parser.add_argument("--acc_cut_off", action = 'store', default = 12 , type = int,
                    help='Cut off frequency for the accelerometer signals [Hz]. Default: 12')
parser.add_argument("--gyro_cut_off", action = 'store', default = 2 , type = int,
                    help='Cut off frequency for the gyrooscope signals [Hz]. Default: 2')
parser.add_argument('--overlap', default=False, action='store_true',
                    help='If you want to use overlap in the data, write True. Default: False')
parser.add_argument("--overlap_size", action = 'store', default = 0.5 , type = float,
                    help='Size of overlap in percent. Float between 0-1. Default 0.5')
parser.add_argument("--block_size", action = 'store', default = 512 , type = int,
                    help='Size of the blocks, in samples. Defalut: 512')

args = parser.parse_args()

#save model to disk as ...
save_as = args.save_as
#cut off frequency of the IR filter
#accelerometer
acc_cut_off = args.acc_cut_off
#gyrooscope
gyro_cut_off = args.gyro_cut_off

#transform data with overlap
overlap = args.overlap
overlap_size = args.overlap_size
block_size  = args.block_size

#pwelch nperseg
nperseg = block_size/2


############################################################################

if __name__ == '__main__':
    print("Create Labels")
    print('*'*20)
    labels = create_labels()

    print("Read Data")
    print('*'*20)
    read_data()
    df = read_data()

    print("Label Data")
    print('*'*20)
    labeled_df = add_activity_label(df, labels)

    print("Filter data")
    print('*'*20)
    filtered_df_acc = filter_acc(labeled_df,cutoff = acc_cut_off)
    filtered_df_gyro = filter_gyro(labeled_df,cutoff= gyro_cut_off)

    labeled_df = remake_df(filtered_df_acc, filtered_df_gyro, labeled_df)

    #labeled_df = drop_unlabeled(labeled_df)

    labeled_df = renindex_df(labeled_df)

    print("Add Blocks")
    print('*'*20)

    if overlap:
        block_df = create_block_df(labeled_df,block_size,overlap_size)
    else:
        block_df = create_block_df_no_overlap(labeled_df,block_size)

    print("Add Activity labels")
    print('*'*20)
    activity_labels = create_activity_labels(block_df)

    print("Aggregate Data")
    print('*'*20)
    agg_df = create_aggregated(block_df)

    fft_df = do_fft(block_df,nperseg=nperseg)

    fft_agg_df = create_aggregated_freq(fft_df)

    print("Add Features")
    print('*'*20)
    features = create_features(agg_df, fft_agg_df)


    print("Drop na-s")
    print('*'*20)
    features_to_drop = find_na(features)
    print(features_to_drop)
    features = drop_features(features,features_to_drop)

    print("ML model")
    print('*'*20)
    X_train, X_test, y_train, y_test = create_train_test(features, activity_labels)

    # print("Drop na-s")
    # print('*'*20)
    # # features_to_drop = find_na(X_train)
    # # X_train = drop_features(X_train,features_to_drop)
    # # X_test = drop_features(X_test,features_to_drop)

    start_train = time.time()
    model = train_model(X_train, y_train)
    train_time = round(time.time()-start_train,1)

    accuracy = accuracy_score(y_test,model.predict(X_test))
    recall = recall_score(y_test,model.predict(X_test), average=None)
    precision = precision_score(y_test,model.predict(X_test), average=None)
    f1 = f1_score(y_test,model.predict(X_test), average=None)


    print('*'*20)
    print(f'Train time: {train_time}s')
    print(f'Accuracy score: {round(accuracy,3)}%')
)
    save_model(name=save_as, model=model)
