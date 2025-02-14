import os
import shutil

# 元のフォルダと、新しいフォルダのパスを指定
source_folder = 'REST database' # 画像が入っている親フォルダ
destination_folder = 'REST_Hands'  # 画像をまとめたい新しいフォルダ

# 新しいフォルダが存在しない場合、作成
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# os.walk()を使って、親ディレクトリとすべての子ディレクトリを再帰的に探索
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        # 画像ファイルだけを選択（例えば、jpgやpng）
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(destination_folder, filename)
            
            # ファイルを新しいフォルダに移動（コピーしたい場合はshutil.copyを使用）
            #shutil.move(source_path, destination_path)
            shutil.copy(source_path, destination_path)  # コピーする場合

print("画像がすべて新しいフォルダにまとめられました。")
