from json import encoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re

# determinacja kolejnoścci w encodingu or some shit like that - to zapewnia stałość encodingu
vocab_top=['-p','+p','-l','+l','d']
vocab_seq=['1','2','3','4']
vocab_classe=['0','1','2','3']

#import csv z zestawem danych
df_train = pd.read_csv('Dataset_secundo_testo/train_set_RF_4c_1.csv')
df_test = pd.read_csv('Dataset_secundo_testo/test_set_RF_4c_1.csv')


#rozdzielenie zestawu danych
X_train = df_train[['top', 'seq']]
Y_train = df_train['cluster_results_KMedoids']
X_test = df_test[['top', 'seq']]
Y_test = df_test['cluster_results_KMedoids']

# coping for sake of art
X_test=X_test.copy()
X_train=X_train.copy()
Y_train=Y_train.copy()
Y_test=Y_test.copy()

#podział danych na pojedyńcze znaki do enkodowania
X_train['split_top'] = X_train['top'].apply(
    lambda text: re.findall(r'[+-][pl]|d', str(text)))
X_train['split_seq']= X_train['seq'].apply(
    lambda text: re.findall(r'1|2|3|4', str(text)))
X_test['split_top'] = X_test['top'].apply(
    lambda text: re.findall(r'[+-][pl]|d', str(text)))
X_test['split_seq']= X_test['seq'].apply(
    lambda text: re.findall(r'1|2|3|4', str(text)))

expended_X_train_pos=pd.DataFrame(X_train['split_top'].tolist(), index=X_train.index)
expended_X_train_pos.columns = [f'Loop_{i+1}' for i in range(expended_X_train_pos.shape[1])]
expended_X_train_seq=pd.DataFrame(X_train['split_seq'].tolist(), index=X_train.index)
expended_X_train_seq.columns=[f'Seq_{i+1}' for i in range(expended_X_train_seq.shape[1])]
expended_X_test_pos=pd.DataFrame(X_test['split_top'].tolist(), index=X_test.index)
expended_X_test_pos.columns = [f'Loop_{i+1}' for i in range(expended_X_test_pos.shape[1])]
expended_X_test_seq=pd.DataFrame(X_test['split_seq'].tolist(), index=X_test.index)
expended_X_test_seq.columns=[f'Seq_{i+1}' for i in range(expended_X_test_seq.shape[1])]
expended_Y_train=pd.DataFrame(Y_train.tolist(), index=Y_train.index)
expended_Y_test=pd.DataFrame(Y_test.tolist(), index=Y_test.index)

#encoding
# Train set - X
pos_list_train=[vocab_top]*expended_X_train_pos.shape[1]
encoder_pos=OneHotEncoder(categories=pos_list_train, sparse_output=False, handle_unknown='ignore')
encoded_pos=encoder_pos.fit_transform(expended_X_train_pos)
encoded_pos_df=pd.DataFrame(encoded_pos, index=X_train.index, columns=encoder_pos.get_feature_names_out())
X_train = pd.concat([X_train, encoded_pos_df], axis=1)
seq_list_train=[vocab_seq]*expended_X_train_seq.shape[1]
encoder_seq=OneHotEncoder(categories=seq_list_train, sparse_output=False, handle_unknown='ignore')
encoded_seq=encoder_seq.fit_transform(expended_X_train_seq)
encoded_seq_df=pd.DataFrame(encoded_seq, index=X_train.index, columns=encoder_seq.get_feature_names_out())
X_train = pd.concat([X_train, encoded_seq_df], axis=1)

# train set - Y
class_list_test=[vocab_classe]*expended_Y_test.shape[1]
encoder_classe=OneHotEncoder(categories=class_list_test, sparse_output=False, handle_unknown='ignore')
encoded_classe=encoder_classe.fit_transform(Y_test.values.reshape(-1,1))
encoded_classe_df=pd.DataFrame(encoded_classe, index=Y_test.index, columns=encoder_classe.get_feature_names_out())
Y_test = pd.concat([Y_test, encoded_classe_df], axis=1)

# test set - X
pos_list_test=[vocab_top]*expended_X_test_pos.shape[1]
encoder_pos=OneHotEncoder(categories=pos_list_test, sparse_output=False, handle_unknown='ignore')
encoded_pos=encoder_pos.fit_transform(expended_X_test_pos)
encoded_pos_df=pd.DataFrame(encoded_pos, index=X_test.index, columns=encoder_pos.get_feature_names_out())
X_test = pd.concat([X_test, encoded_pos_df], axis=1)
seq_list_test=[vocab_seq]*expended_X_test_seq.shape[1]
encoder_seq=OneHotEncoder(categories=seq_list_test, sparse_output=False, handle_unknown='ignore')
encoded_seq=encoder_seq.fit_transform(expended_X_test_seq)
encoded_seq_df=pd.DataFrame(encoded_seq, index=X_test.index, columns=encoder_seq.get_feature_names_out())
X_test = pd.concat([X_test, encoded_seq_df], axis=1)

# test set - Y
class_list_train=[vocab_classe]*expended_Y_train.shape[1]
encoder_classe=OneHotEncoder(categories=class_list_train, sparse_output=False, handle_unknown='ignore')
encoded_classe=encoder_classe.fit_transform(Y_train.values.reshape(-1,1))
encoded_classe_df=pd.DataFrame(encoded_classe, index=Y_train.index, columns=encoder_classe.get_feature_names_out())
Y_train = pd.concat([Y_train, encoded_classe_df], axis=1)


# czyszczenie danych again
X_train=X_train.drop(columns=['top','seq','split_top','split_seq'])
X_test=X_test.drop(columns=['top','seq','split_top','split_seq'])
Y_test=Y_test.drop(columns=['cluster_results_KMedoids'])
Y_train=Y_train.drop(columns=['cluster_results_KMedoids'])
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
Y_test.to_csv('Y_test.csv', index=False)
Y_train.to_csv('Y_train.csv', index=False)
