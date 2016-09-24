clear all;
close all;
clc;
train_interval = 100;
test_interval = 1000;

fid = fopen('train_loss', 'r');
train_loss = fscanf(fid, '%f\n');
fclose(fid);
n = 1 : length(train_loss);
idx_train = n * train_interval;

fid = fopen('test_loss', 'r');
test_loss = fscanf(fid, '%f\n');
fclose(fid);
n = 1 : length(test_loss);
idx_test = n * test_interval;

fid = fopen('test_acc', 'r');
test_acc = fscanf(fid, '%f\n');
fclose(fid);

figure;
plot(idx_train, train_loss);
hold on;
plot(idx_test, test_loss);
grid on;
legend('train_loss', 'test_loss');
xlabel('iterations');
ylabel('loss');
title('train & test loss');

figure;
plot(idx_test, test_acc * 100);
grid on;
xlabel('iterations');
ylabel('accuracy(%)');
title('test accuracy');
