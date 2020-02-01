% プログラム概要：回帰モデルの作成と検証
% 検証方法：LOOCV
% 回帰モデル：Ridge回帰
% データ：タスク1回目と2回目の生理指標の変化量とアンケート指標(説明変数)

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

%% 今は使わないが　特に下の方の統計的な評価の算出は利用するかも

% % % 説明変数の正規化
% % x1 = normalize(x1);
% % x2 = normalize(x2);
% % x3 = normalize(x3);
% % x4 = normalize(x4);
% % x5 = normalize(x5);
% 
% % データの合計数をloopに代入
% loop = size(x1, 1);
% 
% % 変数
% RC = [];
% predict = [];
% 
% % Leave One Outを行うためのfor文
% for i = 1:loop
% 
%     % 全ての説明変数と目的変数のデータを格納
%     trainX = [x1 x2 x3 x4 x5];
%     trainY  = y ;
%     
%     % テストデータの作成
%     testX = trainX(i, :);
%     testY = trainY(i);
%     
%     % テストデータのみの行を削除し、学習データの作成
%     trainX(i, :) = [];
%     trainY(i) = [];
%     
%     % Ridge回帰作成
%     D = x2fx(trainX,'linear'); % 回帰分析のために計画行列へ変換
%     D(:,1) = []; % No constant term （１列目の定数項を削除）
%     k = 0.1; % k = 0:1e-5:5e-3; %0~0.005まで 1e-5ずつステップさせる
%     b = ridge(trainY, D, k, 0); %y:実測値の推定したい値　X:説明変数 k:重み係数λ 0:元のデータスケールに復元 b(0)有り
%     
%     % 回帰係数を格納
%     RC = cat(2, RC, b);
%     
%     % 作成したモデルにテストデータを入れて、推定値を算出
%     yhat = b(1) + testX(:,:) * b(2:end);
%     
%     %推定値を格納
%     predict = cat(1, predict, yhat);
%         
% end 
% 
% % 実測値と推定値のグラフを作成
% scatter(y, predict) % 推定値と実測値の散布図の作成
% hold on
% plot(y,y)
% xlabel('正規化した瞬目数の実測値')
% ylabel('正規化した瞬目数の推定値')
% hold off
% 
% 
% % 実測値と予測値の相関係数
% r = corr2(y, predict);
% 
% % SSE 残差変動の平方和
% zansa = y - predict;
% SSE = sum(zansa.^2);
% 
% % SST 全変動の平方和
% SST_before = y - mean(y);
% SST = sum(SST_before.^2);
% 
% % 予測的説明分散 r^2cvを算出
% r2cv = 1 - ( SSE / SST)
% 
% % MAE 平均絶対誤差
% MAE = sum( abs(zansa) ) / loop
% 
% % MSE 平均二乗誤差
% MSE = SSE / loop
% 
% % RMSE 平均平方二乗誤差
% RMSE = sqrt(MSE)
% 
