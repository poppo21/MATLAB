% プログラム概要：回帰モデルの作成と検証
% 回帰モデル：NN回帰
% データ：タスク1回目と2回目の生理指標の変化量(説明変数)

cd E:\MATLAB\statistics\Learning_1s\data;

bio_data = ["CVRR1", "HR1", "LF_HF1", "LHpa1", "Mpa1", "blink1"]; % 入力ファイル名の配列
data_variety = length(bio_data); % 入力するファイル数の確認

folder_name_def = 'Sub%d'; % 入力するファイルを格納するフォルダ名の定義
file_data_def = 'Sub%d_%s.csv'; % 入力するファイル名の定義

% 変数宣言（for文で配列を結合しているため）
x1 = [];
x2 = [];
x3 = [];
x4 = [];
x5 = [];

y1 = [];
y2 = [];

%% データの読み込み(学習データ)

for user_num = 3 : 14
    folder_name = sprintf(folder_name_def, user_num) ; % 出力ファイルを格納するフォルダ名の設定
    cd (folder_name); % 被験者のデータへアクセス
    
    for i = 1 : data_variety
        file_data = sprintf(file_data_def, user_num, bio_data(i) ); % 指定した入力ファイル名に書き換え
        name{i} = file_data ; % データの名前を格納 　　　　cell配列以外だとエラー（左辺のインデックスが右辺とサイズが適合しないため、代入は実行できません。）
        
    end
    
    % 変数の読み込み csvファイル形式 (説明変数) cat(タスク1回目と2回目のデータをまとめている)
    x1 = cat(1, x1, csvread( name{1}, 0, 0) );
    x2 = cat(1, x2, csvread( name{2}, 0, 0) );
    x3 = cat(1, x3, csvread( name{3}, 0, 0) );
    x4 = cat(1, x4, csvread( name{4}, 0, 0) );
    x5 = cat(1, x5, csvread( name{5}, 0, 0) );
    
    % 変数の読み込み csvファイル形式 (目的変数)
    y2 = cat(1, y2, csvread( name{6}, 0, 0) );
    
    cd ../ ; % 1つ上の階層へ戻る（dataフォルダへ）  

end

%% テストデータの入力

cd E:\MATLAB\statistics\Learning_1s\data;

bio_data = ["CVRR2", "HR2", "LF_HF2", "LHpa2", "Mpa2", "blink2"]; % 入力ファイル名の配列

% 変数宣言（for文で配列を結合しているため）
Test_x1 = [];
Test_x2 = [];
Test_x3 = [];
Test_x4 = [];
Test_x5 = [];

Test_y1 = [];
Test_y2 = [];

%% データの読み込み(テストデータ)

for user_num = 3 : 14
    folder_name = sprintf(folder_name_def, user_num) ; % 出力ファイルを格納するフォルダ名の設定
    cd (folder_name); % 被験者のデータへアクセス
    
    for i = 1 : data_variety
        file_data = sprintf(file_data_def, user_num, bio_data(i) ); % 指定した入力ファイル名に書き換え
        name{i} = file_data ; % データの名前を格納 　　　　cell配列以外だとエラー（左辺のインデックスが右辺とサイズが適合しないため、代入は実行できません。）
        
    end
    
    % 変数の読み込み csvファイル形式 (説明変数) cat(タスク1回目と2回目のデータをまとめている)
    Test_x1 = cat(1, Test_x1, csvread( name{1}, 0, 0) );
    Test_x2 = cat(1, Test_x2, csvread( name{2}, 0, 0) );
    Test_x3 = cat(1, Test_x3, csvread( name{3}, 0, 0) );
    Test_x4 = cat(1, Test_x4, csvread( name{4}, 0, 0) );
    Test_x5 = cat(1, Test_x5, csvread( name{5}, 0, 0) );
    
    % 変数の読み込み csvファイル形式 (目的変数)
    Test_y2 = cat(1, Test_y2, csvread( name{6}, 0, 0) );
    
    cd ../ ; % 1つ上の階層へ戻る（dataフォルダへ）  

end

cd ../ % dataの1つ上の階層へ移動

%% 目的変数の選択（ここで目的変数の値を決定する）
y = y2;
test_y = Test_y2;

%% 学習データとテストデータに用いる値の格納 ＋ 転置

% 学習データ
X = [x1 x2 x3 x4 x5];
X = X.' ;
T  = y ;
T = T.' ;

% テストデータ
testX = [Test_x1 Test_x2 Test_x3 Test_x4 Test_x5];
testX = testX.' ;
testT  = test_y ;
testT = testT.' ;

%% NNの設定

% setdemorandstream(491218382) % 初期重みを固定しない場合は、コメントアウト

net = fitnet(15); % ニューロン数を指定
view(net) 

[net,tr] = train(net,X,T); % おそらくNNに用いるデータを定義している
nntraintool % NNツールボックスを開く

plotperform(tr) % ネットワーク性能のプロット (詳しくはドキュメントで検索)

net.divideParam.trainRatio = 100/100; % 学習データの割合
net.divideParam.valRatio = 0/100; % ネットワークの汎化を検証するデータの割合　→　過適合の発生前に学習を停止させるために使用
net.divideParam.testRatio = 0/100; % テストデータの割合

% tr　→　ネットワーク学習に関する情報をまとめた構造体 
% .testInd　→　テストセットにそれぞれ使用されたデータ点のインデックス
% testX = X(:,tr.testInd); % テストデータの入力データ　→　testX,Y,Tに、例えばタスク２回目のデータを入れると、進捗ありそう
% testT = T(:,tr.testInd); % テストデータの正解データ
% 
% testY = net(testX); % テストデータをNNに入力した際の推定データ
% 
% perf = mse(net,testT,testY) % 平均二乗正規化誤差性能関数

%% テスト

Y = net(testX); % NNのモデル学習で対象となった全データを入力し、推定値を算出

plotregression(testT,Y) % 線形回帰のプロット

e = testT - Y; % 誤差（残差）を算出

% ploterrhist(e) % ヒストグラムを作成

