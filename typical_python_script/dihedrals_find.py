import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from pathlib import Path
import argparse
import gc
from collections import defaultdict


def warun(time_csv):
    data = pd.read_csv(time_csv)
    data['name_dir'] = data['seq'].astype(str) + '_' + data['top'].astype(str) + '_' + data['lay'].astype(str) + '_' + \
                       data['copy'].astype(str)
    return dict(zip(data['name_dir'], data['unf_time']))


def liczu_liczu_bb_chi(input_dir, output_csv, time_dict, n_clusters, n_sasiad):
    master_fold = Path(input_dir)
    target_filename_bb = 'bb_dihedrals.txt'
    target_filename_chi = 'chi_dihedrals.txt'

    group_folders = defaultdict(list)
    for subdir in master_fold.iterdir():
        if subdir.is_dir():
            try:
                seq, top, lay, copy = subdir.name.split("_")
                group_folders[(seq, top)].append((subdir, lay, copy))
            except ValueError:
                print(f" -> POMINIĘTO: Nazwa '{subdir.name}' nie zgadza się ze wzorcem seq_top_lay_copy.")
                continue

    all_selected_bb = []
    all_selected_chi = []
    processed_already = 0

    for (seq, top), folders in group_folders.items():
        comb_data_bb = []
        comb_data_chi = []

        for subdir, lay, copy in folders:
            folder_name = subdir.name
            target_file_bb = subdir / target_filename_bb
            target_file_chi = subdir / target_filename_chi

            # Sprawdzamy czy istnieją oba pliki
            if not target_file_bb.exists() or not target_file_chi.exists():
                continue
            if folder_name not in time_dict:
                continue

            time_val = time_dict[folder_name]

            if time_val > 100000:
                rows = 5000
            else:
                rows = int(time_val / 20)
            if rows <= 0:
                rows = 1

            # Odczyt nagłówków
            with open(target_file_bb, 'r') as file:
                header_line_bb = file.readline()
            column_names_bb = header_line_bb.replace('#', '').split()

            with open(target_file_chi, 'r') as file:
                header_line_chi = file.readline()
            column_names_chi = header_line_chi.replace('#', '').split()

            # Odczyt danych
            try:
                data_bb = pd.read_csv(target_file_bb, sep=r'\s+', skiprows=1, names=column_names_bb, nrows=rows,
                                      comment='#')
                data_chi = pd.read_csv(target_file_chi, sep=r'\s+', skiprows=1, names=column_names_chi, nrows=rows,
                                       comment='#')
            except Exception as e:
                print(f'Błąd odczytu pliku w {folder_name}: {e}')
                continue

            # Upewniamy się, że oba DataFrame'y mają tę samą liczbę wierszy w razie rozbieżności
            min_len = min(len(data_bb), len(data_chi))

            data_bb = data_bb.iloc[:min_len].assign(seq=seq, top=top, lay=lay, copy=copy)
            data_chi = data_chi.iloc[:min_len].assign(seq=seq, top=top, lay=lay, copy=copy)

            comb_data_bb.append(data_bb)
            comb_data_chi.append(data_chi)

        if not comb_data_bb:
            continue

        # Defrakmentacja w pamięci
        group_df_bb = pd.concat(comb_data_bb, ignore_index=True).copy()
        group_df_chi = pd.concat(comb_data_chi, ignore_index=True).copy()

        # KLASTROWANIE TYLKO NA BAZIE KĄTÓW BB
        kolumny_katow_bb = [col for col in group_df_bb.columns if col not in ['time', 'seq', 'top', 'lay', 'copy']]
        radiany_bb = np.radians(group_df_bb[kolumny_katow_bb])

        sining = np.sin(radiany_bb).add_suffix('_sin')
        cosing = np.cos(radiany_bb).add_suffix('_cos')

        cluster_data = pd.concat([sining, cosing], axis=1).astype('float32')

        actual_clusters = min(n_clusters, len(group_df_bb))
        kmedoids = MiniBatchKMeans(n_clusters=actual_clusters, random_state=10, batch_size=4096).fit(cluster_data)

        distances = pairwise_distances(cluster_data, kmedoids.cluster_centers_, metric='euclidean')
        actual_neighbors = min(n_sasiad, len(group_df_bb))

        for c in range(actual_clusters):
            # Wybieramy indeksy na podstawie odległości kątów BB
            indices = np.argsort(distances[:, c])[:actual_neighbors]

            # Wycinamy te same indeksy z BB
            selected_chunk_bb = group_df_bb.iloc[indices].copy()
            selected_chunk_bb = selected_chunk_bb.assign(
                cluster_id=c + 1,
                rank=np.arange(len(indices)),
                og_idx=indices
            )
            all_selected_bb.append(selected_chunk_bb)

            # Wycinamy dokładnie TE SAME indeksy z CHI
            selected_chunk_chi = group_df_chi.iloc[indices].copy()
            selected_chunk_chi = selected_chunk_chi.assign(
                cluster_id=c + 1,
                rank=np.arange(len(indices)),
                og_idx=indices
            )
            all_selected_chi.append(selected_chunk_chi)

        processed_already += 1
        if processed_already % 10 == 0:
            print(f'Skrypt działa i przetworzył grupę: {processed_already}')

        del group_df_bb, group_df_chi, cluster_data, comb_data_bb, comb_data_chi, distances
        gc.collect()

    # Zapis danych dla BB
    if all_selected_bb:
        df_fin_bb = pd.concat(all_selected_bb, ignore_index=True)
        meta_cols = ['top', 'seq', 'lay', 'copy', 'cluster_id', 'rank', 'og_idx', 'time']
        meta_cols_exist_bb = [c for c in meta_cols if c in df_fin_bb.columns]
        other_cols_bb = [c for c in df_fin_bb.columns if c not in meta_cols_exist_bb]
        df_fin_bb = df_fin_bb.reindex(columns=meta_cols_exist_bb + other_cols_bb)

        df_fin_bb.to_csv(f'{output_csv}_bb.csv', index=False)
        print(f'Zakończono wybieranie medoidów i zapisano w pliku: {output_csv}_bb.csv')
    else:
        print("Nie znaleziono żadnych danych do przetworzenia dla BB.")

    # Zapis powiązanych danych dla CHI
    if all_selected_chi:
        df_fin_chi = pd.concat(all_selected_chi, ignore_index=True)
        meta_cols_exist_chi = [c for c in meta_cols if c in df_fin_chi.columns]
        other_cols_chi = [c for c in df_fin_chi.columns if c not in meta_cols_exist_chi]
        df_fin_chi = df_fin_chi.reindex(columns=meta_cols_exist_chi + other_cols_chi)

        # Opcjonalnie usuwamy og_idx z pliku CHI, by zachować spójność z poprzednią wersją skryptu
        if 'og_idx' in df_fin_chi.columns:
            df_fin_chi = df_fin_chi.drop('og_idx', axis=1)

        df_fin_chi.to_csv(f'{output_csv}_chi.csv', index=False)
        print(f'Zakończono zapis powiązanych kątów w pliku: {output_csv}_chi.csv')
    else:
        print("Nie znaleziono żadnych danych do przetworzenia dla CHI.")


if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('-i', '--input_dir', type=str, required=True, help='Katalog z danymi wejściowymi')
    paser.add_argument('-o', '--output_csv', type=str, required=True, help='Prefiks dla plików wyjściowych')
    paser.add_argument('-c', '--condition_csv', type=str, required=True, help='Plik CSV z warunkami czasowymi')

    paser.add_argument('-k', '--clusters', type=int, default=1, help='Liczba klastrów')
    paser.add_argument('-n', '--neighbors', type=int, default=5, help='Liczba struktur do wybrania')

    args = paser.parse_args()

    time_dict = warun(args.condition_csv)

    # Pojedyncze wywołanie funkcji przetwarzającej oba typy kątów
    liczu_liczu_bb_chi(args.input_dir, args.output_csv, time_dict, args.clusters, args.neighbors)