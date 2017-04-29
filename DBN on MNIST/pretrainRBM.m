%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : pretrainRBM.m                                       %
%  Description      : Pre-training the restricted boltzmann machine (RBM) %
%                     model by Contrastive Divergence Learning CD-1       %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function rbm = pretrainRBM(rbm, V, Configurations )

% If parameters does excist copy them from main
if( exist('Configurations' ) )
 if( isfield(Configurations,'MaxIter') )
  MaxIter = Configurations.MaxIter;
 end
 if( isfield(Configurations,'InitialMomentum') )
  InitialMomentum = Configurations.InitialMomentum;
 end
 if( isfield(Configurations,'InitialMomentumIter') )
  InitialMomentumIter = Configurations.InitialMomentumIter;
 end
 if( isfield(Configurations,'FinalMomentum') )
  FinalMomentum = Configurations.FinalMomentum;
 end
 if( isfield(Configurations,'ComPromOut') )
  ComPromOut = Configurations.ComPromOut;
 end
 if( isfield(Configurations,'NormFact') )
  NormFact = Configurations.NormFact;
 end
 if( isfield(Configurations,'BatchSize') )
  BatchSize = Configurations.BatchSize;
 end
else
 Configurations = [];
end

num = size(V,1);           % Input database patterns dimensions
dimH = size(rbm.b, 2);     % Hiden nodes dimensions
dimV = size(rbm.c, 2);     % Vissible nodes dimensions

% If the batch size is not declared take all the patterns as one batch
if( BatchSize <= 0 )
  BatchSize = num;
end

deltaW = zeros(dimV, dimH);     % Increasing factor of vissible to hidden units weights
deltaB = zeros(1, dimH);        % Increasing factor of hidden units biases
deltaC = zeros(1, dimV);        % Increasing factor of vissible units biases

if( ComPromOut ) 
	% Start time measure
    timer = tic;
end

for iter=1:MaxIter

    % Set momentum to prevent the system from converging to a local minimum
	if( iter <= InitialMomentumIter )
		momentum = InitialMomentum;
	else
		momentum = FinalMomentum;
    end  
	ind = randperm(num);   
	for batch=1:BatchSize:num

		bind = ind(batch:min([batch + BatchSize - 1, num]));
        
        % =================== POSSITIVE PHASE ===================== %      
        % Possitive phase - activation probabillity of the hidden units P(h|v)
        % Gibbs sampling step 0.Using CD (Contrastive Divergence) - 1 step
        vis0 = double(V(bind,:)); % Set patterns of visible nodes
        % Mathematical function [6.13]
        hid0 = v2h( rbm, vis0 );  % Compute hidden nodes from the vissible nodes
        % =================== END OF POSSITIVE PHASE ============= %
        
        % =================== NEGATIVE PHASE ===================== %
        % Negative phase - activation probabillity of the vissible units P(v|h)
        % Gibbs sampling step 1.Using CD (Contrastive Divergence) - 1 step
        bhid0 = double( rand(size(hid0)) < hid0 ); 
        % Mathematical function [6.13]
        rec = h2v( rbm, bhid0 );    % Compute vissible nodes from the hidden nodes
        % Mathematical function [6.14]
        hid1 = v2h( rbm, rec );   % Compute hidden nodes from the vissible nodes again [CD-1]
        
        posprods = hid0' * vis0; % Possitive phase statistics - <vihj>0
        negprods = hid1' * rec; % Negative phase statistics - <vihj>1
        % =================== END OF NEGATIVE PHASE ================= % 
		
        % =================== UPDATE WEIGHTS AND BIASES ============= %   		
		% Compute the weights update by contrastive divergence		
        dW = (posprods - negprods)';
        dB = (sum(hid0, 1) - sum(hid1, 1));
        dC = (sum(vis0, 1) - sum(rec, 1));
        
        % For Gaussian - Bernoulli kernel
        if( strcmpi( 'GBRBM', rbm.type ) )
        	dW = bsxfun(@rdivide, dW, rbm.sig');
        	dC = bsxfun(@rdivide, dC, rbm.sig .* rbm.sig);
        end
        
        % Compute update parameters (vissible bias- hidden bias- Weights)
		deltaW = momentum * deltaW + (NormFact / num) * dW;
		deltaB = momentum * deltaB + (NormFact / num) * dB;
		deltaC = momentum * deltaC + (NormFact / num) * dC;
        
		% Vissible - Hidden weights update % Mathematical function [6.17]
		rbm.W = rbm.W + deltaW - rbm.W;
		rbm.b = rbm.b + deltaB; % Hidden biases update
		rbm.c = rbm.c + deltaC; % Vissible biases update

    end
        % =================== END OF UPDATES ============= %
    
    % Output pre - training information to command prompt 
	if( ComPromOut )
        H = v2h( rbm, V );                          % Vissible to hidden
        Vr = h2v( rbm, H );                         % hidden to Vissible
		err = power( V - Vr, 2 );                   % Extract reconstruction error
		rmse = sqrt( sum(err(:)) / numel(err) );    % Root mean square error
        
        % Time and iterations informations
        totalti = toc(timer);
        aveti = totalti / iter;
        estti = (MaxIter-iter) * aveti;
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');    
		fprintf( ' iter %3d : MSE %3.4f : All Batches time %3.4f : Remaining Time : %s\n', iter, rmse, aveti, eststr );
    end
end

