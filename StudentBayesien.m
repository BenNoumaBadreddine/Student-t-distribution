%__Badreddine
%Leave one out avant de modéliser les distributions de données, 
%Classifier l'élément tiré en utilisant la règle de bayes sur les distributions de student 

clear
clc
Seed=33;
randn('seed',Seed);
rand('seed',Seed);
% 
filenameA = '/Users/badr/Desktop/database/data1.txt';
filenameB = '/Users/badr/Desktop/database/data2.txt';
filenameC = '/Users/badr/Desktop/database/data3.txt';
 
delimiterIn = ' ';

A = importdata(filenameA,delimiterIn);
B = importdata(filenameB,delimiterIn);
C = importdata(filenameC,delimiterIn);
 
A=A(1:21,:);
B=B(1:21,:);
C=C(1:21,:);
%___________________________________________________________________________________________
 %Modélisation des classes avec la famille de distribution T-Student distribution
 %___________________________________________________________________________________________

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
