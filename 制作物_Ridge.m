% �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�����ϐ�)
x1 = csvread('CVRR1.csv', 0, 0);
x2 = csvread('HR1.csv', 0, 0);
x3 = csvread('LF_HF1.csv', 0, 0);
x4 = csvread('LHpa1.csv', 0, 0);
x5 = csvread('Mpa1.csv', 0, 0);
x6 = csvread('RTLX.csv', 0, 0);
x7 = csvread('BIS_BAS.csv', 0, 0);

% �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�ړI�ϐ�)
y1 = csvread('Touch1.csv', 0, 0);
y2 = csvread('blink1_delta.csv', 0, 0);
y3 = csvread('blink1_gensho_per.csv', 0, 0);

% �V�����ړI�ϐ� ( [���^�b�`���̐��K�������l] + [�u�ڐ��̕ϓ����̐��K�������l] )
y4 = normalize(y1) + normalize(y3);

% �ړI�ϐ��̑I���i�����ŖړI�ϐ��̒l�����肷��j
y = y4;

% �����ϐ��̐��K��
x1 = normalize(x1);
x2 = normalize(x2);
x3 = normalize(x3);
x4 = normalize(x4);
x5 = normalize(x5);
x6 = normalize(x6);
x7 = normalize(x7);

% �팱�Ґ���human�ɑ��
human = size(x1, 1);

% �ϐ�
RC = [];
predict = [];

% Leave One Out���s�����߂�for��
for i = 1:human

    % �S�Ă̐����ϐ��ƖړI�ϐ��̃f�[�^���i�[
    trainX = [x1 x2 x3 x4 x5 x6 x7];
    trainY  = y ;
    
    % �e�X�g�f�[�^�̍쐬
    testX = trainX(i, :);
    testY = trainY(i);
    
    % �e�X�g�f�[�^�݂̂̍s���폜���A�w�K�f�[�^�̍쐬
    trainX(i, :) = [];
    trainY(i) = [];
    
    % Ridge��A�쐬
    D = x2fx(trainX,'linear'); % ��A���͂̂��߂Ɍv��s��֕ϊ�
    D(:,1) = []; % No constant term �i�P��ڂ̒萔�����폜�j
    k = 0.1; % k = 0:1e-5:5e-3; %0~0.005�܂� 1e-5���X�e�b�v������
    b = ridge(trainY, D, k, 0); %y:�����l�̐��肵�����l�@X:�����ϐ� k:�d�݌W���� 0:���̃f�[�^�X�P�[���ɕ��� b(0)�L��
    
    % ��A�W�����i�[
    RC = cat(2, RC, b);
    
    % �쐬�������f���Ƀe�X�g�f�[�^�����āA����l���Z�o
    yhat = b(1) + testX(:,:) * b(2:end);
    
    %����l���i�[
    predict = cat(1, predict, yhat);
        
end 

% �����l�Ɛ���l�̃O���t���쐬
scatter(y, predict) % ����l�Ǝ����l�̎U�z�}�̍쐬
hold on
plot(y,y)
xlabel('���^�b�`���̎����l')
ylabel('���^�b�`���̐���l')
hold off


% �����l�Ɨ\���l�̑��֌W��
r = corr2(y, predict);

% SSE �c���ϓ��̕����a
zansa = y - predict;
SSE = sum(zansa.^2);

% SST �S�ϓ��̕����a
SST_before = y - mean(y);
SST = sum(SST_before.^2);

% �\���I�������U r^2cv���Z�o
r2cv = 1 - ( SSE / SST)

% MSE ���ϓ��덷
MSE = SSE / human

% RMSE ���ϕ������덷
RMSE = sqrt(MSE)

