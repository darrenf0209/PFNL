% Super-resolution scale
scale=2;
% Testing dataset directory
fpath = 'C:\Users\darre\OneDrive\Documents\GitHub\PFNL\test\udm10\';
% Model name
model='alt_only_cur_downsize_20200904_output_feedback_20200909';
frames_foregone = 5;

fid=fopen(strcat(fpath,model,'.txt'),'wt');
fprintf(fid,'{\n');
% Retrieve list of testing video sets
list0=dir(strcat(fpath));
idx=0;
sum_psnr=0;
sum_ssim=0;
for j = 1:length(list0)
    if( isequal( list0( j ).name, '.' )||...
        isequal( list0( j ).name, '..')||...
        ~list0( j ).isdir)               
        continue;
    end
    idx=idx+1;
    kind=strcat(fpath,list0(j).name);
    % Truth video frame directory
    path1=strcat(kind,'/truth_downsize_2_frames_foregone/');
    % Model video frame direcotry
    path2=strcat(kind,'/',model,'/');
    % Retrieve list of all images for each directory
    list1= dir(strcat(path1,'*.png'));
    list2= dir(strcat(path2,'*.png'));
    sum_p=0;
    sum_s=0;
    index=0;
    fprintf(fid,strcat('\"',list0(j).name,'\": {\n'));
    fprintf(fid,'\"frame\": [\n{');
    % Skip the first and last frame
    for i=2:length(list1)-1
%         fprintf('%d\n', i + frames_foregone - 1)
        index=index+1;
        img1_name=list1(i).name;
        img1=imread(strcat(path1,img1_name));
        img2_name=list2(i).name;
        img2=imread(strcat(path2,img2_name));
        % Crop 8px border
        boundarypixels = 8; 
        img1 = img1(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
        img2 = img2(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
        % Convert images RGB to YCbCr
        if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
        end
        if size(img2, 3) == 3
            img2 = rgb2ycbcr(img2);
            img2 = img2(:, :, 1);
        end
        % Computer PSNR for image
        psnr=compute_psnr(img1,img2,scale);
        sum_p=sum_p+psnr;
%         fprintf('%d %f\n',i,psnr);
        ssim=SSIM(img1,img2);
        sum_s=sum_s+ssim; 
%         fprintf(fid, strcat('\"%d\":[%g,%g]'),i + frames_foregone - 1, psnr, ssim);
        if i == length(list1)-2
            fprintf(fid, strcat('\"%d\":[%g,%g]'),i + frames_foregone - 1, psnr, ssim);
        else
            fprintf(fid, strcat('\"%d\":[%g,%g],'),i + frames_foregone - 1, psnr, ssim);
        end
%         fprintf('%d %f\n',i,ssim);
    end
    avg_p=sum_p/index;
    avg_s=sum_s/index;
    fprintf(fid,'}\n],\n');
    fprintf(fid,strcat('\"average\":[%g,%g]\n},\n'),avg_p,avg_s);
    fprintf(list0(j).name);
    fprintf('\n%f\n',avg_p);
    fprintf('%f\n',avg_s);
    sum_psnr=sum_psnr+avg_p;
    sum_ssim=sum_ssim+avg_s;
end
average_p=sum_psnr/idx;
average_s=sum_ssim/idx;
fprintf('%f\n',average_p);
fprintf('%f',average_s);
fprintf(fid,strcat('\"average of average\":[%g,%g]\n}'),average_p,average_s);
