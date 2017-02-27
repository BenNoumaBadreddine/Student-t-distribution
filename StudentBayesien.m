%__Badreddine
%Modélisation des classes avec la famille de distribution T-Student
%distribution

clear
clc
Seed=33;
randn('seed',Seed);
rand('seed',Seed);
% 
filenameA = '/Users/badr/Desktop/PSO/database/Origine FR, FT, FT+FR/OA_FR.txt';
filenameB = '/Users/badr/Desktop/PSO/database/Origine FR, FT, FT+FR/OA_FT.txt';
filenameC = '/Users/badr/Desktop/PSO/database/Origine FR, FT, FT+FR/OA_FR_FT.txt';
% 
% filenameA = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Region_OA_FR.txt';
% filenameB = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Region_OA_FT.txt';
% filenameC = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Region_OA_FR_FT.txt';
% % % 
%  filenameA = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Mean_OA_FR_Valid.txt';
%  filenameB = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Mean_OA_FT_Valid.txt';
%  filenameC = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/Mean_OA_FR_FT_Valid.txt';
% % 
% % 
% filenameA = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/OA FR/Corrected_OA_FR.txt';
% filenameB = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/OA FT/Corrected_OA_FT.txt';
% filenameC = '/Users/badr/Desktop/PSO/database/Database 20-06-2016/OA FR FT/Corrected_OA_FR_FT.txt';

delimiterIn = ' ';

A = importdata(filenameA,delimiterIn);
B = importdata(filenameB,delimiterIn);
C = importdata(filenameC,delimiterIn);
CC = importdata(filenameC,delimiterIn);
BB = importdata(filenameB,delimiterIn);

A=A(1:21,:);
B=B(1:21,:);
C=C(1:21,:);
C(4,:)=CC(33,:);
%C(2,:)=CC(23,:);
C(3,:)=CC(24,:);
C(5,:)=CC(26,:);
C(6,:)=CC(27,:);
C(7,:)=CC(29,:);
C(8,:)=CC(28,:);
C(9,:)=CC(41,:);
C(10,:)=CC(51,:);
C(11,:)=CC(36,:);
C(12,:)=CC(46,:);
C(16,:)=CC(37,:);
C(17,:)=CC(40,:);
C(18,:)=CC(55,:);
C(19,:)=CC(57,:);
C(21,:)=CC(54,:);
 
fprintf('Please Wait Preparing Data Class A ...\n');
[fittedA1c,fittedA1S,fittedA1nu]=fitt(A);
fprintf('Please Wait Preparing Data Class B ...\n');
[fittedB1c,fittedB1S,fittedB1nu]=fitt(B);
fprintf('Please Wait Preparing Data Class C ...\n');
[fittedC1c,fittedC1S,fittedC1nu]=fitt(C);

DataBase=[A;B;C];
Class=[ones(size(A,1),1);ones(size(B,1),1)*2;ones(size(C,1),1)*3];

ClassPredict=[];
fittedA1nu=20;
fittedB1nu=20;
fittedC1nu=20;        
        for i=1:size(DataBase,1)
            
            TestData=DataBase(i,:);
            TestClass = Class(i);
            
            TrainingData=DataBase(1:end~=i,:);
            TrainingClass=Class(1:end~=i,:);
               
            IdxA=find(TrainingClass==1);
            IdxB=find(TrainingClass==2);
            IdxC=find(TrainingClass==3);
            
                     
            switch (TestClass)
                case 1
                    [fittedA1c,fittedA1S,fittedA1nu]=fitt(TrainingData(IdxA,:)); 
                    %fittedA1nu=19;
 
                case 2
                    [fittedB1c,fittedB1S,fittedB1nu]=fitt(TrainingData(IdxB,:));
                     %fittedB1nu=19;
                case 3
                    [fittedC1c,fittedC1S,fittedC1nu]=fitt(TrainingData(IdxC,:));
                     %fittedC1nu=19;
            end
        
            val1=mvtpdff(TestData,fittedA1S,fittedA1nu);
            val2=mvtpdff(TestData,fittedB1S,fittedB1nu);
            val3=mvtpdff(TestData,fittedC1S,fittedC1nu);
            [val,indexe]=max([val1,val2,val3]);
            
 
            switch (indexe)
                case 1
                    ClassPredict(i)=1;
                case 2
                    ClassPredict(i)=2;
                case 3
                    ClassPredict(i)=3;
            end
            
        end
        
        TauxBayes = sum(ClassPredict'==[ones(size(A,1),1);ones(size(B,1),1)*2;ones(size(C,1),1)*3])/size(Class,1)*100;
        
        
        fprintf('Leave-One-Out avec : %.0f%% \n' ,TauxBayes);
        ConfusionMatrix = confusionmat([ones(size(A,1),1);ones(size(B,1),1)*2;ones(size(C,1),1)*3],ClassPredict)


%%
