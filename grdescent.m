%implements Geoffreys Hinton Rprop, the batch learning analogue of RMS
%Prop

function [w]=grdescent_rprop(func,w0,stepsize,maxiter,tolerance)




if nargin<5
    tolerance=1e-02;
end
  

StopCriteria=0;
iter=0;
signs=ones(length(w0),1); 
stepsizes= stepsize.*ones(length(w0),1); 



while(~StopCriteria)
    iter=iter+1;
    [loss,gradient]=func(w0);
    
    %norm(gradient) %uncomment if you want to watch the gradient decrease
    
    if(norm(gradient)<tolerance || iter>maxiter)
        StopCriteria=1;
        w=w0;
    else
        
        idx=signs==sign(gradient);   % check to see if sign of gradient matches last iteration
        stepsizes(idx)= stepsizes(idx).*1.2; % increase gradient for those with matching signs
        stepsizes(~idx)=stepsizes(~idx).*.5; %decrease gradient for those mismatching
        stepsizes= min(stepsizes,50);   % limit step size to a max
        stepsizes= max(stepsizes,.000001);  %and a min 
        
        w0= w0-stepsizes.*gradient;
        signs=sign(gradient);
        
    end
    
end

%iter

end
