tic;
FN_ReadMNIST;


test_labels_idx=9;
test_images_idx=17;

x_test=reshape(test_images(test_images_idx:(test_images_idx+784*10000-1)),dx,dx,10000);
x_test=permute(x_test,[2 1 3]);
x_test=round(x_test./max(x_test(:)))./256./0.03;

%load labels
t_y=zeros(10,10000);
t_y(test_labels((test_labels_idx:test_labels_idx+10000-1))+(0:10:10*(10000-1))'+1)=1;
%convolution
a1_in_test=zeros(da1,da1,nf,10000);
for jj=1:da1
    for kk=1:da1
        a1_in_test(jj,kk,:,:)=(reshape(x_test(jj:(jj+df-1),kk:(kk+df-1),:),df^2,10000)'*reshape(f1r,df^2,nf))'+repmat(f_b',1,10000);
    end
end
%activation
a1_test=max(0,a1_in_test);
%BN_FWD
%pooling
[p1_test,p1idx_test]=max(reshape(permute(reshape(a1_test,da1,dp,[]),[2 1 3]),dp^2,[]));
p1_test=reshape(p1_test,dap,dap,nf,10000);
%fully connected network
p1r_test=reshape(p1_test,dap^2*nf,10000);
p1r_test=[p1r_test; ones(1,10000)];
fc_in_test=wfc'*p1r_test;
%activation
fc_test=max(0,fc_in_test);
fc_test=[fc_test;ones(1,10000)];
%output layer
y_in_test=wy'*fc_test;
%activation
y_test=max(0,y_in_test);
%{
%y_in=max(0,y_in);
%BN_FWD
[y_norm,y]=FN_BN_TEST(y_in,gma_y,bta_y,avr_g_y,var_g_y,0);
avr_g_y=[avr_g_y avr_y];
var_g_y=[var_g_y var_y];
%}
%output analysis and error function
test_correct=sum(sum(((y_test-max(y_test)+eps)>0).*(y_test>0).*t_y));

test_yield=test_correct/10000*100;

toc;
    
optstr=['recognition accuracy:', num2str(test_yield), '%'];
disp(optstr)

%{
for i=1:20
figure;
subplot(2,6,1);
imagesc(x_test(:,:,i));
subplot(2,6,2);
imagesc(a1_test(:,:,1,i));
subplot(2,6,3);
imagesc(a1_test(:,:,2,i));
subplot(2,6,4);
imagesc(a1_test(:,:,3,i));
subplot(2,6,5);
imagesc(a1_test(:,:,4,i));
subplot(2,6,6);
plot([0:9],y_test(:,i));
title('Inference Result');

end
%}