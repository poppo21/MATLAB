cd E:\MATLAB\statistics\LOOCV_1s\data;

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

%% �f�[�^�̓ǂݍ���

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

cd ../ % data��1��̊K�w�ֈړ�

%% �ړI�ϐ��̑I���i�����ŖړI�ϐ��̒l�����肷��j
y = y2;

% NN�̐ݒ�
%% �w�K�f�[�^�ɗp����l�̊i�[ �{ �]�u
X = [x1 x2 x3 x4 x5];
X = X.' ;
T  = y ;
T = T.' ;

%% NN�̐ݒ�

% setdemorandstream(491218382) % �����d�݂��Œ肵�Ȃ��ꍇ�́A�R�����g�A�E�g

net = fitnet(15); % �j���[���������w��
view(net) 

[net,tr] = train(net,X,T); % �����炭NN�ɗp����f�[�^���`���Ă���
nntraintool % NN�c�[���{�b�N�X���J��

plotperform(tr) % �l�b�g���[�N���\�̃v���b�g (�ڂ����̓h�L�������g�Ō���)

net.divideParam.trainRatio = 50/100; % �w�K�f�[�^�̊���
net.divideParam.valRatio = 20/100; % �l�b�g���[�N�̔ĉ������؂���f�[�^�̊����@���@�ߓK���̔����O�Ɋw�K���~�����邽�߂Ɏg�p
net.divideParam.testRatio = 30/100; % �e�X�g�f�[�^�̊���

% tr�@���@�l�b�g���[�N�w�K�Ɋւ�������܂Ƃ߂��\���� 
% .testInd�@���@�e�X�g�Z�b�g�ɂ��ꂼ��g�p���ꂽ�f�[�^�_�̃C���f�b�N�X
testX = X(:,tr.testInd); % �e�X�g�f�[�^�̓��̓f�[�^�@���@testX,Y,T�ɁA�Ⴆ�΃^�X�N�Q��ڂ̃f�[�^������ƁA�i�����肻��
testT = T(:,tr.testInd); % �e�X�g�f�[�^�̐����f�[�^

testY = net(testX); % �e�X�g�f�[�^��NN�ɓ��͂����ۂ̐���f�[�^

perf = mse(net,testT,testY) % ���ϓ�搳�K���덷���\�֐�

Y = net(X); % NN�̃��f���w�K�őΏۂƂȂ����S�f�[�^����͂��A����l���Z�o

plotregression(T,Y) % ���`��A�̃v���b�g

e = T - Y; % �덷�i�c���j���Z�o

ploterrhist(e) % �q�X�g�O�������쐬

