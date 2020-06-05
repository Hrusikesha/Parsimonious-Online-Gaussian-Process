function [op_vector]=runningaverage(inputvector,running_sample_size)
ip_length=length(inputvector);
%rem=mod(ip_length,running_sample_size);
%no_of_op_samples=fix(ip_length/running_sample_size);
%inputvector(ip_length+1:ip_length+running_sample_size-rem)=inputvector(ip_length);

for j=1:(ip_length-running_sample_size+1)
    op_vector(j)=sum(inputvector(j:j+running_sample_size-1))/running_sample_size;
end
end
