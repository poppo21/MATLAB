% 変数の読み込み csvファイル形式 (説明変数)
x1 = csvread('CVRR1.csv', 0, 0);
x2 = csvread('HR1.csv', 0, 0);
x3 = csvread('LF_HF1.csv', 0, 0);
x4 = csvread('LHpa1.csv', 0, 0);
x5 = csvread('Mpa1.csv', 0, 0);
x6 = csvread('RTLX.csv', 0, 0);
x7 = csvread('BIS_BAS.csv', 0, 0);

% 変数の読み込み csvファイル形式 (目的変数)
y1 = csvread('Touch1.csv', 0, 0);
y2 = csvread('blink1_delta.csv', 0, 0);
y3 = csvread('blink1_gensho_per.csv', 0, 0);

% 新しい目的変数 ( [総タッチ数の正規化した値] + [瞬目数の変動率の正規化した値] )
y4 = normalize(y1) + normalize(y3);

% 目的変数の選択（ここで目的変数の値を決定する）
y = y4;

% 説明変数の正規化
x1 = normalize(x1);
x2 = normalize(x2);
x3 = normalize(x3);
x4 = normalize(x4);
x5 = normalize(x5);
x6 = normalize(x6);
x7 = normalize(x7);

% 被験者数をhumanに代入
human = size(x1, 1);

% 変数
RC = [];
predict = [];

% Leave One Outを行うためのfor文
for i = 1:human

    % 全ての説明変数と目的変数のデータを格納
    trainX = [x1 x2 x3 x4 x5 x6 x7];
    trainY  = y ;
    
    % テストデータの作成
    testX = trainX(i, :);
    testY = trainY(i);
    
    % テストデータのみの行を削除し、学習データの作成
    trainX(i, :) = [];
    trainY(i) = [];
    
    % Ridge回帰作成
    D = x2fx(trainX,'linear'); % 回帰分析のために計画行列へ変換
    D(:,1) = []; % No constant term （１列目の定数項を削除）
    k = 0.1; % k = 0:1e-5:5e-3; %0~0.005まで 1e-5ずつステップさせる
    b = ridge(trainY, D, k, 0); %y:実測値の推定したい値　X:説明変数 k:重み係数λ 0:元のデータスケールに復元 b(0)有り
    
    % 回帰係数を格納
    RC = cat(2, RC, b);
    
    % 作成したモデルにテストデータを入れて、推定値を算出
    yhat = b(1) + testX(:,:) * b(2:end);
    
    %推定値を格納
    predict = cat(1, predict, yhat);
        
end 

% 実測値と推定値のグラフを作成
scatter(y, predict) % 推定値と実測値の散布図の作成
hold on
plot(y,y)
xlabel('総タッチ数の実測値')
ylabel('総タッチ数の推定値')
hold off


% 実測値と予測値の相関係数
r = corr2(y, predict);

% SSE 残差変動の平方和
zansa = y - predict;
SSE = sum(zansa.^2);

% SST 全変動の平方和
SST_before = y - mean(y);
SST = sum(SST_before.^2);

% 予測的説明分散 r^2cvを算出
r2cv = 1 - ( SSE / SST)

% MSE 平均二乗誤差
MSE = SSE / human

% RMSE 平均平方二乗誤差
RMSE = sqrt(MSE)

