%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                   in the field of computer vision                       %
%  File             : FineTuning.m                                        %
%  Description      : training the Deep Belief Nets (DBN) model by        %
%                     back projection algorithm                           %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function [dbn rmse] = FineTuning( dbn, IN, OUT, Configurations)


% Important parameters
strbm = 1;
nrbm = numel( dbn.rbm );

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
    if( isfield(Configurations,'NormFact') )
        NormFact = Configurations.NormFact;
    end
    if( isfield(Configurations,'BatchSize') )
        BatchSize = Configurations.BatchSize;
    end
    if( isfield(Configurations,'ComPromOut') )
        ComPromOut = Configurations.ComPromOut;
    end
    if( isfield(Configurations,'HLayer') )
        Layer = Configurations.HLayer;
    end
end

num = size(IN,1); % Size of database input patterns

% If the batch size is not declared take all the patterns as one batch
if( BatchSize <= 0 )
    BatchSize = num;
end

if( Layer > 0 )
    strbm = nrbm - Layer + 1; % Hidden layers
end

deltaDbn = dbn;     % deltaDBN owns the update parameters from backpropagation
for n=strbm:nrbm    % Initialize deltaDBN
    deltaDbn.rbm{n}.W = zeros(size(dbn.rbm{n}.W));
    deltaDbn.rbm{n}.b = zeros(size(dbn.rbm{n}.b));
end

if( Layer > 0 )
    strbm = nrbm - Layer + 1;
end

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
    ind = randperm(num); % Random index for the data batch
    
    % Start fine - tuning using backpropagation
    for batch=1:BatchSize:num
        bind = ind(batch:min([batch + BatchSize - 1, num]));
        
        trainDBN = dbn; % Copy the DBN to a train class
        
        Hall = v2hall( trainDBN, IN(bind,:) ); % Bottom - up. Feed forward the information
        
        for n=nrbm:-1:strbm % Top - down. Backpropagate the error
            derSgm = Hall{n} .* ( 1 - Hall{n} ); % Derivative of the sigmoids
            
            if( n+1 > nrbm )
                der = ( Hall{nrbm} - OUT(bind,:) ); % n layer error
            else
                der = derSgm .* ( der * trainDBN.rbm{n+1}.W' );
            end
            
            if( n-1 > 0 )
                in = Hall{n-1};
            else
                in = IN(bind, :);
            end
            
            in = cat(2, ones(numel(bind),1), in);
            deltaWb = in' * der / numel(bind);
            deltab = deltaWb(1,:);
            deltaW = deltaWb(2:end,:);
            
            if( strcmpi( dbn.rbm{n}.type, 'GBRBM' ) )
                deltaW = bsxfun( @rdivide, deltaW, trainDBN.rbm{n}.sig' );
            end
            
            deltaDbn.rbm{n}.W = momentum * deltaDbn.rbm{n}.W;
            deltaDbn.rbm{n}.b = momentum * deltaDbn.rbm{n}.b;
            deltaDbn.rbm{n}.W = deltaDbn.rbm{n}.W - NormFact * deltaW;
            deltaDbn.rbm{n}.b = deltaDbn.rbm{n}.b - NormFact * deltab;
        end
        
        for n=strbm:nrbm
            dbn.rbm{n}.W = dbn.rbm{n}.W + deltaDbn.rbm{n}.W;
            dbn.rbm{n}.b = dbn.rbm{n}.b + deltaDbn.rbm{n}.b;
        end
        
    end
    
    if( ComPromOut )
        tdbn = dbn;
        out = v2h( tdbn, IN );
        err = power( OUT - out, 2 );
        rmse = sqrt( sum(err(:)) / numel(err) );
        msg = sprintf('%3d : rmse %9.4f', iter, rmse );
        
        totalti = toc(timer);
        aveti = totalti / iter;
        estti = (MaxIter-iter) * aveti;
        eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');
        
        fprintf( 'iter %s Remaining Time : %s\n', msg, eststr );
    end
end
end