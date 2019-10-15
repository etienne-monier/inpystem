function [Dictionary,output] = wKSVD(...
    Data,... % a dXN matrix that contains N signals (Y), each of dimension d.
    Mask, ... % a dXN matrix that contains N corruptions, to be applied to the data
    param)
% =========================================================================
%                         wKSVD algorithm
% =========================================================================
% This file is a modification of the original K-SVD algorithm 
% (update in the function I_findBetterDictionaryElement() ) 
% available on M. Elad's webpage.
% Modifications were introduced to account for erasures/corruptions
% specified by the mask in the dictionary learning phase and are based on 
% "Sparse representation for color image restoration"
% written by J. Mairal, M. Elad, and G. Sapiro, 
% IEEE Trans. on Image Processing, vol. 17, pp. 53-69, 2008.
% =========================================================================
%                         K-SVD algorithm
% =========================================================================
% The K-SVD algorithm finds a dictionary for linear representation of
% signals. Given a set of signals, it searches for the best dictionary that
% can sparsely represent each signal. Detailed discussion on the algorithm
% and possible applications can be found in "The K-SVD: An Algorithm for 
% Designing of Overcomplete Dictionaries for Sparse Representation", written
% by M. Aharon, M. Elad, and A.M. Bruckstein and appeared in the IEEE Trans. 
% On Signal Processing, Vol. 54, no. 11, pp. 4311-4322, November 2006. 
% =========================================================================
% INPUT ARGUMENTS:
% Data    an dXN matrix that contains N signals (Y), each of dimension d.
% Mask    an dXN matrix that contains N corruptions, each of dimension d.
% param   structure including all required parameters for K-SVD execution.
%         Required fields are:
%    K, ...           the number of dictionary elements to train
%    numIteration,... number of iterations to perform.
%    dSparsity,...    sparsity level in OMPm.m 
%                     Note that this version only works for a fixed 
%                     sparsity level and not for a fixed approximation error
%    preserveDCAtom...if =1 then the first atom in the dictionary
%                        is set to be constant, and does not ever change. This
%                        Useful e.g. for natural images 
%                        (in this case, only param.K-1 atoms are trained). 
%    InitializationMethod,...  method to initialize the dictionary, can
%                              be one of the following arguments: 
%                  * 'DataElements' (initialization by the signals themselves), or: 
%                  * 'GivenMatrix' (initialization by a given matrix param.initialDictionary).
%    initialDictionary ((optional, see InitializationMethod) ,...if the initialization method 
%                                 is 'GivenMatrix', this is the matrix that will be used.
%    TrueDictionary (optional), ... if specified, in each
%                                 iteration the difference between this dictionary and the trained one
%                                 is measured and displayed.
%    displayProgress, ...      if =1 progress information = the average 
%                              representation error (RMSE) is displayed.
% =========================================================================
% OUTPUT ARGUMENTS:
%  Dictionary                  The extracted dictionary of size nX(param.K).
%  output                      Struct that contains information about the current run. It may include the following fields:
%    CoefMatrix                  The final coefficients matrix (it should hold that Data equals approximately Dictionary*output.CoefMatrix.
%    ratio                       If the true dictionary was defined (in
%                                synthetic experiments), this parameter holds a vector of length
%                                param.numIteration that includes the detection ratios in each
%                                iteration).
%    totalerr                    The total representation error after each
%                                iteration (defined only if
%                                param.displayProgress=1)                   
% =========================================================================


%%%% preparations

if (~isfield(param,'InitializationMethod'))
    param.InitializationMethod = 'DataElements';
elseif (~isfield(param,'initialDictionary'))
    param.InitializationMethod = 'DataElements';
end


if (~isfield(param,'displayProgress'))
    param.displayProgress = 0;
end

if (~isfield(param,'Xref'))
    param.Xref = 0;
end

totalerr(1) = 99999;

if (isfield(param,'TrueDictionary'))
    displayErrorWithTrueDictionary = 1;
    ErrorBetweenDictionaries = zeros(param.numIteration+1,1);
    ratio = zeros(param.numIteration+1,1);
else
    displayErrorWithTrueDictionary = 0;
	ratio = 0;
end
if (param.preserveDCAtom>0)
    FixedDictionaryElement(1:size(Data,1),1) = 1/sqrt(size(Data,1));
else
    FixedDictionaryElement = [];
end

if (size(Data,2) < param.K)
    disp('Size of data is smaller than the dictionary size. Trivial solution...');
    Dictionary = Data(:,1:size(Data,2));
    return;
elseif (strcmp(param.InitializationMethod,'DataElements'))
    Dictionary(:,1:param.K-param.preserveDCAtom) = Data(:,1:param.K-param.preserveDCAtom);
elseif (strcmp(param.InitializationMethod,'GivenMatrix'))
    % Dictionary(:,1:param.K-param.preserveDCAtom) = param.initialDictionary(:,param.preserveDCAtom+1:1+param.K);
    Dictionary = param.initialDictionary;
end
% reduce the components in Dictionary spanned by the fixed element
if (param.preserveDCAtom)
    tmpMat = FixedDictionaryElement \ Dictionary;
    Dictionary = Dictionary - FixedDictionaryElement*tmpMat;
end
% (ensure) normalisation of dictionary
Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
Dictionary = Dictionary.*repmat(sign(Dictionary(1,:)),size(Dictionary,1),1); % multiply in the sign of the first element.
output.totalerr = zeros(1,param.numIteration);

%%%%% the K-SVD algorithm starts here:
start_t = tic;
for iterNum = 1:param.numIteration
    %%%% sparse approximation using OMPm with fixed sparsity level
    CoefMatrix = OMPm([FixedDictionaryElement,Dictionary],Data, Mask, param.dSparsity);
    %%%% dictionary update
    replacedVectorCounter = 0;
	rPerm = randperm(size(Dictionary,2));
    for j = rPerm
        [betterDictionaryElement,CoefMatrix,addedNewVector] = I_findBetterDictionaryElement(Data, Mask,...
            [FixedDictionaryElement,Dictionary],j+size(FixedDictionaryElement,2),...
            CoefMatrix ,param.dSparsity);
        Dictionary(:,j) = betterDictionaryElement;
        if (param.preserveDCAtom)
            tmpCoef = FixedDictionaryElement\betterDictionaryElement;
            Dictionary(:,j) = betterDictionaryElement - FixedDictionaryElement*tmpCoef;
            Dictionary(:,j) = Dictionary(:,j)./sqrt(Dictionary(:,j)'*Dictionary(:,j));
        end
        replacedVectorCounter = replacedVectorCounter+addedNewVector;
    end

    time_t = toc(start_t);
    if (iterNum>1 & param.displayProgress)

        tmp_out = [FixedDictionaryElement,Dictionary]*CoefMatrix;
%        save(['/home/emonier/Job_FS/data_tmp/' num2str(iterNum) '.mat'], 'tmp_out');

        if numel(param.Xref) ==1
            output.totalerr(iterNum-1) = sqrt(sum(sum((Mask.*(Data-[FixedDictionaryElement,Dictionary]*CoefMatrix)).^2))/prod(size(Data)));
        else
            output.totalerr(iterNum-1) = sum(sum((Mask.*(param.Xref-[FixedDictionaryElement,Dictionary]*CoefMatrix)).^2))/sum(sum(param.Xref.^2));
        end

        disp(['Iteration #',num2str(iterNum),' over ' num2str(param.numIteration) ' (estimated remaining time: ' sec2str(time_t*(param.numIteration-iterNum+1)) '), total error is: ',num2str(output.totalerr(iterNum-1))]);
    end
    start_t = tic;

    if (displayErrorWithTrueDictionary ) 
        [ratio(iterNum+1),ErrorBetweenDictionaries(iterNum+1)] = I_findDistanceBetweenDictionaries(param.TrueDictionary,Dictionary);
        disp(strcat(['Iteration  ', num2str(iterNum),' ratio of restored elements: ',num2str(ratio(iterNum+1))]));
        output.ratio = ratio;
    end
    Dictionary = I_clearDictionary(Dictionary,CoefMatrix(size(FixedDictionaryElement,2)+1:end,:),Data);
    
end

output.CoefMatrix = CoefMatrix;
Dictionary = [FixedDictionaryElement,Dictionary];


%%%%%% auxiliary functions %%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findBetterDictionaryElement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [betterDictionaryElement,CoefMatrix,NewVectorAdded] = I_findBetterDictionaryElement(Data,Mask, Dictionary,j,CoefMatrix,numCoefUsed)
if (length(who('numCoefUsed'))==0)
    numCoefUsed = 1;
end
relevantDataIndices = find(CoefMatrix(j,:)); % data indices that uses the j'th dictionary element.
if (length(relevantDataIndices)<1) % replacement strategy 
    ErrorMat = Data-Dictionary*CoefMatrix;
    ErrorNormVec = sum(ErrorMat.^2);
    [d,i] = max(ErrorNormVec);
    betterDictionaryElement = Data(:,i);%%% other possibility: ErrorMat(:,i); 
    betterDictionaryElement = betterDictionaryElement./sqrt(betterDictionaryElement'*betterDictionaryElement);
    betterDictionaryElement = betterDictionaryElement.*sign(betterDictionaryElement(1));
    CoefMatrix(j,:) = 0;
    NewVectorAdded = 1;
    return;
end

NewVectorAdded = 0;
tmpCoefMatrix = CoefMatrix(:,relevantDataIndices); 
tmpCoefMatrix(j,:) = 0;% the coefficients of the element we now improve are not relevant.
% vector of errors that we want to minimize with the new element
errors =(Data(:,relevantDataIndices) - Dictionary*tmpCoefMatrix); 

%%%% wKSVD update: min || beta.*(errors - atom*coeff) ||_F^2 for beta = mask
iterN = 10;  %%% paper suggests  10-20 (10 works fine and is faster)
betterDictionaryElementNew  = sparse(size(Dictionary,1),1); %%% zero initialisation
CoefMatrixNew = sparse(1, size(relevantDataIndices,2)); %%% zero initialisation
for i=1:iterN
   NewF = Mask(:,relevantDataIndices).*errors + (ones(size(Mask,1),size(relevantDataIndices,2)) - Mask(:,relevantDataIndices)).*(betterDictionaryElementNew*CoefMatrixNew);
   %keyboard 
   [betterDictionaryElementNew,singularValue,betaVector] = svds(NewF,1);
   sign_atom = sign(betterDictionaryElementNew(1, 1));
   betterDictionaryElementNew = betterDictionaryElementNew * sign_atom;
   betaVector = betaVector * sign_atom;
   
   CoefMatrixNew = singularValue*betaVector';
end
betterDictionaryElement = betterDictionaryElementNew;
CoefMatrix(j,relevantDataIndices) = CoefMatrixNew;

%%%% for comparsion ksvd update:
%%% [betterDictionaryElement,singularValue,betaVector] = svds(errors,1);
%%% CoefMatrix(j,relevantDataIndices) = singularValue*betaVector';
                                                            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  findDistanceBetweenDictionaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ratio,totalDistances] = I_findDistanceBetweenDictionaries(original,new)
% first, all the column in original starts with positive values.
catchCounter = 0;
totalDistances = 0;
for i = 1:size(new,2)
    new(:,i) = sign(new(1,i))*new(:,i);
end
for i = 1:size(original,2)
    d = sign(original(1,i))*original(:,i);
    distances =sum ( (new-repmat(d,1,size(new,2))).^2);
    [minValue,index] = min(distances);
    errorOfElement = 1-abs(new(:,index)'*d);
    totalDistances = totalDistances+errorOfElement;
    catchCounter = catchCounter+(errorOfElement<0.01);
end
ratio = 100*catchCounter/size(original,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  I_clearDictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Dictionary = I_clearDictionary(Dictionary,CoefMatrix,Data)
T2 = 0.99;
T1 = 3;
K=size(Dictionary,2);
Er=sum((Data-Dictionary*CoefMatrix).^2,1); % remove identical atoms
G=Dictionary'*Dictionary; G = G-diag(diag(G));
for jj=1:1:K,
    if max(G(jj,:))>T2 | length(find(abs(CoefMatrix(jj,:))>1e-7))<=T1 ,
        [val,pos]=max(Er);
        Er(pos(1))=0;
        Dictionary(:,jj)=Data(:,pos(1))/norm(Data(:,pos(1)));
        G=Dictionary'*Dictionary; G = G-diag(diag(G));
    end;
end;

