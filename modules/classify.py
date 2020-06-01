from func_data_preparation import *
from func_ml import *
import warnings
warnings.filterwarnings(action='once')

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
filtered_df_acc = filter_acc(labeled_df,cutoff = 12)
filtered_df_gyro = filter_gyro(labeled_df,cutoff= 2)

labeled_df = remake_df(filtered_df_acc, filtered_df_gyro, labeled_df)

labeled_df = drop_unlabeled(labeled_df)

labeled_df = renindex_df(labeled_df)

print("Add Blocks")
print('*'*20)
#block_df = create_block_df(labeled_df,1024,0.5)
block_df = create_block_df_no_overlap(labeled_df,128)

print("Add Activity labels")
print('*'*20)
activity_labels = create_activity_labels(block_df)

print("Aggregate Data")
print('*'*20)
agg_df = create_aggregated(block_df)

fft_df = do_fft(block_df,nperseg=64)

fft_agg_df = create_aggregated_freq(fft_df)

print("Add Features")
print('*'*20)
features = create_features(agg_df, fft_agg_df)


print("Drop na-s")
print('*'*20)
features_to_drop = find_na(features)
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
model = init_rfc(X_train, y_train)
train_time = round(time.time()-start_train,1)

accuracy = accuracy_score(y_test,model.predict(X_test))
recall = recall_score(y_test,model.predict(X_test), average=None)
precision = precision_score(y_test,model.predict(X_test), average=None)
f1 = f1_score(y_test,model.predict(X_test), average=None)

# print('*'*20)
# print(f'Scores of {model}:')
print('*'*20)
print(f'Train time: {train_time}s')
print(f'Accuracy score: {accuracy}')
# print('\n')
# print(f'Recall score: {recall}')
# print('\n')
# print(f'Precision score: {precision}')
# print('\n')
# print(f'F1 score: {f1}')
# print('\n')
save_model(name='../models/har_model_v07.sav', model=model)
