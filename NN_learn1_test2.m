% �v���O�����T�v�F��A���f���̍쐬�ƌ���
% ���ؕ��@�FLOOCV
% ��A���f���FRidge��A
% �f�[�^�F�^�X�N1��ڂ�2��ڂ̐����w�W�̕ω��ʂƃA���P�[�g�w�W(�����ϐ�)

cd E:\MATLAB\statistics\Learning_1s\data;

bio_data = ["CVRR1", "HR1", "LF_HF1", "LHpa1", "Mpa1", "blink1"]; % ���̓t�@�C�����̔z��
data_variety = length(bio_data); % ���͂���t�@�C�����̊m�F

folder_name_def = 'Sub%d'; % ���͂���t�@�C�����i�[����t�H���_���̒�`
file_data_def = 'Sub%d_%s.csv'; % ���͂���t�@�C�����̒�`

% �ϐ��錾�ifor���Ŕz����������Ă��邽�߁j
x1 = [];
x2 = [];
x3 = [];
x4 = [];
x5 = [];

y1 = [];
y2 = [];

%% �f�[�^�̓ǂݍ���(�w�K�f�[�^)

for user_num = 3 : 14
    folder_name = sprintf(folder_name_def, user_num) ; % �o�̓t�@�C�����i�[����t�H���_���̐ݒ�
    cd (folder_name); % �팱�҂̃f�[�^�փA�N�Z�X
    
    for i = 1 : data_variety
        file_data = sprintf(file_data_def, user_num, bio_data(i) ); % �w�肵�����̓t�@�C�����ɏ�������
        name{i} = file_data ; % �f�[�^�̖��O���i�[ �@�@�@�@cell�z��ȊO���ƃG���[�i���ӂ̃C���f�b�N�X���E�ӂƃT�C�Y���K�����Ȃ����߁A����͎��s�ł��܂���B�j
        
    end
    
    % �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�����ϐ�) cat(�^�X�N1��ڂ�2��ڂ̃f�[�^���܂Ƃ߂Ă���)
    x1 = cat(1, x1, csvread( name{1}, 0, 0) );
    x2 = cat(1, x2, csvread( name{2}, 0, 0) );
    x3 = cat(1, x3, csvread( name{3}, 0, 0) );
    x4 = cat(1, x4, csvread( name{4}, 0, 0) );
    x5 = cat(1, x5, csvread( name{5}, 0, 0) );
    
    % �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�ړI�ϐ�)
    y2 = cat(1, y2, csvread( name{6}, 0, 0) );
    
    cd ../ ; % 1��̊K�w�֖߂�idata�t�H���_�ցj  

end

%% �e�X�g�f�[�^�̓���

cd E:\MATLAB\statistics\Learning_1s\data;

bio_data = ["CVRR2", "HR2", "LF_HF2", "LHpa2", "Mpa2", "blink2"]; % ���̓t�@�C�����̔z��

% �ϐ��錾�ifor���Ŕz����������Ă��邽�߁j
Test_x1 = [];
Test_x2 = [];
Test_x3 = [];
Test_x4 = [];
Test_x5 = [];

Test_y1 = [];
Test_y2 = [];

%% �f�[�^�̓ǂݍ���(�e�X�g�f�[�^)

for user_num = 3 : 14
    folder_name = sprintf(folder_name_def, user_num) ; % �o�̓t�@�C�����i�[����t�H���_���̐ݒ�
    cd (folder_name); % �팱�҂̃f�[�^�փA�N�Z�X
    
    for i = 1 : data_variety
        file_data = sprintf(file_data_def, user_num, bio_data(i) ); % �w�肵�����̓t�@�C�����ɏ�������
        name{i} = file_data ; % �f�[�^�̖��O���i�[ �@�@�@�@cell�z��ȊO���ƃG���[�i���ӂ̃C���f�b�N�X���E�ӂƃT�C�Y���K�����Ȃ����߁A����͎��s�ł��܂���B�j
        
    end
    
    % �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�����ϐ�) cat(�^�X�N1��ڂ�2��ڂ̃f�[�^���܂Ƃ߂Ă���)
    Test_x1 = cat(1, Test_x1, csvread( name{1}, 0, 0) );
    Test_x2 = cat(1, Test_x2, csvread( name{2}, 0, 0) );
    Test_x3 = cat(1, Test_x3, csvread( name{3}, 0, 0) );
    Test_x4 = cat(1, Test_x4, csvread( name{4}, 0, 0) );
    Test_x5 = cat(1, Test_x5, csvread( name{5}, 0, 0) );
    
    % �ϐ��̓ǂݍ��� csv�t�@�C���`�� (�ړI�ϐ�)
    Test_y2 = cat(1, Test_y2, csvread( name{6}, 0, 0) );
    
    cd ../ ; % 1��̊K�w�֖߂�idata�t�H���_�ցj  

end

cd ../ % data��1��̊K�w�ֈړ�

%% �ړI�ϐ��̑I���i�����ŖړI�ϐ��̒l�����肷��j
y = y2;
test_y = Test_y2;

%% �w�K�f�[�^�ƃe�X�g�f�[�^�ɗp����l�̊i�[ �{ �]�u

% �w�K�f�[�^
X = [x1 x2 x3 x4 x5];
X = X.' ;
T  = y ;
T = T.' ;

% �e�X�g�f�[�^
testX = [Test_x1 Test_x2 Test_x3 Test_x4 Test_x5];
testX = testX.' ;
testT  = test_y ;
testT = testT.' ;

%% NN�̐ݒ�

% setdemorandstream(491218382) % �����d�݂��Œ肵�Ȃ��ꍇ�́A�R�����g�A�E�g

net = fitnet(15); % �j���[���������w��
view(net) 

[net,tr] = train(net,X,T); % �����炭NN�ɗp����f�[�^���`���Ă���
nntraintool % NN�c�[���{�b�N�X���J��

plotperform(tr) % �l�b�g���[�N���\�̃v���b�g (�ڂ����̓h�L�������g�Ō���)

net.divideParam.trainRatio = 100/100; % �w�K�f�[�^�̊���
net.divideParam.valRatio = 0/100; % �l�b�g���[�N�̔ĉ������؂���f�[�^�̊����@���@�ߓK���̔����O�Ɋw�K���~�����邽�߂Ɏg�p
net.divideParam.testRatio = 0/100; % �e�X�g�f�[�^�̊���

% tr�@���@�l�b�g���[�N�w�K�Ɋւ�������܂Ƃ߂��\���� 
% .testInd�@���@�e�X�g�Z�b�g�ɂ��ꂼ��g�p���ꂽ�f�[�^�_�̃C���f�b�N�X
% testX = X(:,tr.testInd); % �e�X�g�f�[�^�̓��̓f�[�^�@���@testX,Y,T�ɁA�Ⴆ�΃^�X�N�Q��ڂ̃f�[�^������ƁA�i�����肻��
% testT = T(:,tr.testInd); % �e�X�g�f�[�^�̐����f�[�^
% 
% testY = net(testX); % �e�X�g�f�[�^��NN�ɓ��͂����ۂ̐���f�[�^
% 
% perf = mse(net,testT,testY) % ���ϓ�搳�K���덷���\�֐�

%% �e�X�g

Y = net(testX); % NN�̃��f���w�K�őΏۂƂȂ����S�f�[�^����͂��A����l���Z�o

plotregression(testT,Y) % ���`��A�̃v���b�g

e = testT - Y; % �덷�i�c���j���Z�o

% ploterrhist(e) % �q�X�g�O�������쐬

%% ���͎g��Ȃ����@���ɉ��̕��̓��v�I�ȕ]���̎Z�o�͗��p���邩��

% % % �����ϐ��̐��K��
% % x1 = normalize(x1);
% % x2 = normalize(x2);
% % x3 = normalize(x3);
% % x4 = normalize(x4);
% % x5 = normalize(x5);
% 
% % �f�[�^�̍��v����loop�ɑ��
% loop = size(x1, 1);
% 
% % �ϐ�
% RC = [];
% predict = [];
% 
% % Leave One Out���s�����߂�for��
% for i = 1:loop
% 
%     % �S�Ă̐����ϐ��ƖړI�ϐ��̃f�[�^���i�[
%     trainX = [x1 x2 x3 x4 x5];
%     trainY  = y ;
%     
%     % �e�X�g�f�[�^�̍쐬
%     testX = trainX(i, :);
%     testY = trainY(i);
%     
%     % �e�X�g�f�[�^�݂̂̍s���폜���A�w�K�f�[�^�̍쐬
%     trainX(i, :) = [];
%     trainY(i) = [];
%     
%     % Ridge��A�쐬
%     D = x2fx(trainX,'linear'); % ��A���͂̂��߂Ɍv��s��֕ϊ�
%     D(:,1) = []; % No constant term �i�P��ڂ̒萔�����폜�j
%     k = 0.1; % k = 0:1e-5:5e-3; %0~0.005�܂� 1e-5���X�e�b�v������
%     b = ridge(trainY, D, k, 0); %y:�����l�̐��肵�����l�@X:�����ϐ� k:�d�݌W���� 0:���̃f�[�^�X�P�[���ɕ��� b(0)�L��
%     
%     % ��A�W�����i�[
%     RC = cat(2, RC, b);
%     
%     % �쐬�������f���Ƀe�X�g�f�[�^�����āA����l���Z�o
%     yhat = b(1) + testX(:,:) * b(2:end);
%     
%     %����l���i�[
%     predict = cat(1, predict, yhat);
%         
% end 
% 
% % �����l�Ɛ���l�̃O���t���쐬
% scatter(y, predict) % ����l�Ǝ����l�̎U�z�}�̍쐬
% hold on
% plot(y,y)
% xlabel('���K�������u�ڐ��̎����l')
% ylabel('���K�������u�ڐ��̐���l')
% hold off
% 
% 
% % �����l�Ɨ\���l�̑��֌W��
% r = corr2(y, predict);
% 
% % SSE �c���ϓ��̕����a
% zansa = y - predict;
% SSE = sum(zansa.^2);
% 
% % SST �S�ϓ��̕����a
% SST_before = y - mean(y);
% SST = sum(SST_before.^2);
% 
% % �\���I�������U r^2cv���Z�o
% r2cv = 1 - ( SSE / SST)
% 
% % MAE ���ϐ�Ό덷
% MAE = sum( abs(zansa) ) / loop
% 
% % MSE ���ϓ��덷
% MSE = SSE / loop
% 
% % RMSE ���ϕ������덷
% RMSE = sqrt(MSE)
% 
