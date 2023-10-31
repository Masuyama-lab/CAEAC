function output = CAEAC_Test(testlabel, lastlabel, EstLabel)

if isempty(lastlabel) == 1
    ACC = nan;
    NMI = nan;
    microF = nan;
    macroF = nan;
    ARI = nan;
else

% Compute Accuracy
[ACC, ~, ~, ~, ~, ~, ~] = Evaluate(EstLabel, testlabel);

% Compute Mutual Information
[NMI, ~] = NormalizedMutualInformation( testlabel, EstLabel );

% Compute Micro- Macro- F-Measure
[ microF, macroF ] = MicroMacroFScore( testlabel, EstLabel );

% Compute Adjusted Rand Index
ARI = AdjustedRandIndex( testlabel, EstLabel );

end

output.ACC = ACC;
output.NMI = NMI;
output.microF = microF;
output.macroF = macroF;
output.ARI = ARI;
end


function [normMI, MI] = NormalizedMutualInformation(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   normMI: normalized mutual information normMI=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));


% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
normMI = max(0,z);

end


function [ microFscore, macroFscore ] = MicroMacroFScore(ACTUAL, PREDICTED)
%computer micro and macro: precision, recall and fscore
%Sandy wltongxing@163.com
%micro>macro?
mat=confusionmat(ACTUAL, PREDICTED);
%label_unique=unique([orig_label(:);pred_label(:)]);
%     microTP=0;
%     microFP=0;
%     microFN=0;
len=size(mat,1);
macroTP=zeros(len,1);
macroFP=zeros(len,1);
macroFN=zeros(len,1);
macroP=zeros(len,1);
macroR=zeros(len,1);
macroF=zeros(len,1);
for i=1:len
    macroTP(i)=mat(i,i);
    macroFP(i)=sum(mat(:, i))-mat(i,i);
    macroFN(i)=sum(mat(i,:))-mat(i,i);
    macroP(i)=macroTP(i)/(macroTP(i)+macroFP(i));
    macroR(i)=macroTP(i)/(macroTP(i)+macroFN(i));
    macroF(i)=2*macroP(i)*macroR(i)/(macroP(i)+macroR(i));
end

macroF(isnan(macroF))=0; % NaN to zero
macroR(isnan(macroR))=0;
macroP(isnan(macroP))=0;
macro.precision=mean(macroP);
macro.recall=mean(macroR);
macro.fscore=mean(macroF);


micro.precision=sum(macroTP)/(sum(macroTP)+sum(macroFP));
micro.recall=sum(macroTP)/(sum(macroTP)+sum(macroFN));
micro.fscore=2*micro.precision*micro.recall/(micro.precision+micro.recall);
micro.fscore(isnan(micro.fscore))=0; % NaN to zero
micro.recall(isnan(micro.recall))=0;
micro.precision(isnan(micro.precision))=0;

microFscore = micro.fscore;
macroFscore = macro.fscore;

end


function ARI = AdjustedRandIndex(ACTUAL, PREDICTED)

%function adjrand=adjrand(u,v)
%
% Computes the adjusted Rand index to assess the quality of a clustering.
% Perfectly random clustering returns the minimum score of 0, perfect
% clustering returns the maximum score of 1.
%
%INPUTS
% u = the labeling as predicted by a clustering algorithm
% v = the true labeling
%
%OUTPUTS
% adjrand = the adjusted Rand index
%
%
%Author: Tijl De Bie, february 2003.

n=length(PREDICTED);
ku=max(PREDICTED);
kv=max(ACTUAL);
m=zeros(ku,kv);
for i=1:n
    m(PREDICTED(i),ACTUAL(i))=m(PREDICTED(i),ACTUAL(i))+1;
end
mu=sum(m,2);
mv=sum(m,1);

a=0;
for i=1:ku
    for j=1:kv
        if m(i,j)>1
            a=a+nchoosek(m(i,j),2);
        end
    end
end

b1=0;
b2=0;
for i=1:ku
    if mu(i)>1
        b1=b1+nchoosek(mu(i),2);
    end
end
for i=1:kv
    if mv(i)>1
        b2=b2+nchoosek(mv(i),2);
    end
end

c=nchoosek(n,2);

ARI=(a-b1*b2/c)/(0.5*(b1+b2)-b1*b2/c);

if ARI<0
    ARI = 0;
end


end


function [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean] = Evaluate(PREDICTED, ACTUAL)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);

 end

