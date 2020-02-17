% program to perform ex situ training of a convolutional neural network
% with one convolution and pooling layer
% author: peng lin
% contact: linp@mit.edu

FN_ReadMNIST;
%
%Setup Parameters
dx=28;      %size of input images
nf=4;       %number of filters
df=3;       %size of the filters
S=1;        %stride of the image (currently needs to set to 1)
dp=2;      %size of pooling
dfc=200;    %size of the 1st FC layers
dy=10;      %size of the output
mc=0.95;    %momentum coefficient
%eta2=.00005;
%eta3=.00005;
%
 
FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 700, 700]);
                
eta1=.001;
correct=0;
yield_old=0;

Ep=0;
Ep_old=0;
Ep_log=[];
Eta_log=[];
E_log_x=[];           

%Initialize the array
x=double(rand(dx,dx));                  %input array
f1=double(rand(df,df,nf)/df/df);        %filter array
f_b=double(rand(1,nf));               %filter bias
delta_f1=double(zeros(df,df,nf));
delta_f_b=double(zeros(1,nf));
delta_f1_old=double(zeros(df,df,nf));
delta_f_b_old=double(zeros(1,nf));

da1=double((dx-df)/S+1);                %size of the 1st Conv ouput
a1=double(rand(da1,da1,nf));            %1st conv layer output
delta_a1=double(zeros(da1,da1,nf));

dap=double(da1/dp);                     %size of pooling output
p1=double(rand(dap,dap,nf));              %1st pool layer output

wfc=double(rand(dap^2*nf+1,dfc)/dap^2/nf); %1st FC weigtht matrix
delta_wfc_old=double(zeros(dap^2*nf+1,dfc));
delta_wfc=double(zeros(dap^2*nf+1,dfc));
fc=double(rand(1,dfc));                 %1st FC layer output

wy=double(rand(dfc+1,dy)/dfc/dy);       %Output weigtht matrix
delta_wy_old=double(zeros(dfc+1,dy));
delta_wy=double(zeros(dfc+1,dy));
y=double(rand(1,dy));                   %Output
t_y=double(rand(1,dy));                 %Training Set Output
yield_array=[];

%}
flag=0;
plot_ii=[];
yield_array=[];

h = figure(1);
axis tight manual % this ensures that getframe() returns a consistent size

for epoch=1:20
    train_labels_idx=9;
    train_images_idx=17;
    if (flag==1)
        continue;
    end
    for ii=1:60000
        if (flag==1)
            continue;
        end
        %process input training sample
        x=reshape(train_images(train_images_idx:(train_images_idx+783)),[28 28])';
        %x = [zeros(1,30);zeros(28,1),x zeros(28,1);zeros(1,30)];
        x=round(x./max(x(:)))./256./0.03;

        t_y=0.1.*ones([1 10]);
        t_y(train_labels(train_labels_idx)+1)=0.9;

        train_images_idx = train_images_idx + 784;
        train_labels_idx = train_labels_idx + 1;

        %convolution
        f1r=round(f1./max(abs(f1))); % binarize the kernel
        for i=1:da1
            for j=1:da1
                a1(i,j,:)=reshape(x(i:(i+df-1),j:(j+df-1)),1,[])*reshape(f1r,[],nf)+f_b;
            end
        end
        a1=max(0,a1);
        %pooling
        [p1,p1idx]=max(reshape(permute(reshape(a1,da1,dp,[]),[2 1 3]),dp^2,[]));
        p1=reshape(p1,dap,dap,nf);
        delta_a1=zeros(dp^2,length(p1idx));
        delta_a1(p1idx+(0:dp^2:(length(p1idx)-1)*dp^2))=1;
        delta_a1=reshape(permute(reshape(delta_a1,dp,da1,[]),[2 1 3]),da1,da1,nf);
        %fully connected network
        p1r=reshape(p1,1,dap^2*nf);
        p1r=[p1r 1];
        fc=p1r*wfc;
        fc=max(0,fc);
        fc=[fc 1];
        y=fc*wy;  
        y=max(0,y);

        %output analysis and error function
        opt=find(y==max(y));
        if (t_y(opt)==0.9)
            correct = correct+1;
        end
        EF=0.5*sum((t_y-y).^2);
        Ep=Ep+EF;

        %back propogation
        delta_y=(t_y-y).*(y>0);
        delta_w=eta1*fc'*delta_y + mc.*delta_wy_old;
        wy=wy+delta_w;
        delta_wy_old=delta_w;

        delta_fc=(fc>0).*(delta_y*wy');
        delta_fc=delta_fc(1:dfc);
        delta_w=eta1*p1r'*delta_fc + mc.*delta_wfc_old;
        wfc=wfc+ delta_w;
        delta_wfc_old=delta_w;

        delta_p1=(p1r>0).*(delta_fc*wfc');
        delta_p1=reshape(delta_p1(1:dap^2*nf),[dap,dap,nf]);
        delta_a1=delta_a1.*(repelem(delta_p1,dp,dp));

        %convolution
        delta_f1_old=delta_f1;
        delta_f_b_old=delta_f_b;
        delta_f1=zeros(df,df,nf);
        for i=1:df
            for j=1:df
                delta_f1(i,j,:)=reshape(x(i:(i+da1-1),j:(j+da1-1)),1,[])*reshape(delta_a1,[],nf);
            end
        end

        delta_f1=delta_f1+mc.*delta_f1_old;
        delta_f_b=sum(delta_a1(:)) + mc.*delta_f_b_old;

        f1=f1+eta1*delta_f1;
        f_b=f_b+eta1*delta_f_b;

        delta_f1_old=delta_f1;
        delta_f_b_old=delta_f_b;

        if (mod(ii,500)<1)
            Eta_log=[Eta_log eta1];
            Ep_log=[Ep_log Ep];
            E_log_x=[E_log_x (epoch-1)*60000+ii];

            if Ep/Ep_old<.98
                if eta1<3e-3
                    eta1=1.3*eta1;
                end
            elseif Ep/Ep_old>1.015
                eta1=0.8*eta1;
            end
            Ep_old=Ep;
            Ep=0;
        end

        if (mod(ii,1000)<1)

            yield=correct/10;
            correct=0;
            plot_ii=[plot_ii (epoch-1)*60000+ii];
            yield_array=[yield_array yield];

            optstr= ['iteration:',num2str(ii),'/60000, epoch:',num2str(epoch),'/50,accuracy:',num2str(yield),'%'];
            disp(optstr);
            %disp(f1);
            %disp(f_b);
            figure(1);
            subplot(5,3,1);
            plot(plot_ii,yield_array,'-x');
            title('current accuracy');

            subplot(5,3,4);
            image(reshape(x*256,[dx dx]));
            title('Image');
            subplot(5,3,5);
            bar(0:9,y);
            hold on;
            tmpidx = find(y==max(y),1);
            bar(tmpidx-1,y(tmpidx),'FaceColor',[0.85,0.33,0.1]);
            hold off;
            ylim([0,1]);
            title('Neuron Output');
            subplot(5,3,6);
            bar(0:9,t_y>0.5,'FaceColor',[0.85,0.33,0.1]);
            title('Label');

            subplot(5,4,[9,10,13,14,11,15]);
            imagesc(wfc);
            title('wfc');

            subplot(5,4,[12,16]);
            imagesc(wy);
            title('wy');

            subplot(5,3,2);
            plot(E_log_x,log10(Eta_log),'-x');
            title('Eta');

            subplot(5,3,3);
            plot(E_log_x,log10(Ep_log),'-x');
            title('Ep');

            subplot(5,4,17);
            imagesc(f1(:,:,1));
            title('kernel#1');

            subplot(5,4,18);
            imagesc(f1(:,:,2));
            title('kernel#2');

            subplot(5,4,19);
            imagesc(f1(:,:,3));
            title('kernel#3');

            subplot(5,4,20);
            imagesc(f1(:,:,4));
            title('kernel#4');

            drawnow;                       

        end

        if (mod(ii,20000)<1)
            ex_situ_cnn_test;
        end
    end
end
 