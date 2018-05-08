clear all;

tic
cnt =27;
for i=1
    cnt = cnt + 1;
    runname = ['BOOK-0824731-0', num2str(cnt,'%03d'),'.jpg'];
    RGB = imread(runname);
    gray = rgb2gray(RGB);
    bw = im2bw(gray,0.94);
    wb = ~bw;
    
       
    st = regionprops(wb,'Image');
    cnt2=0;
    for k = 1 : length(st)
        image = st(k).Image;
        sz= size(image);
        sz1 = sz(1);
        sz2 = sz(2);
        if sz1>= 200 & sz1<= 2000 & sz2 >= 150
%             filledImage = imfill(image,'holes');
            J=imresize(image,[200 200]);
            J = ~J;
            cnt2= cnt2 + 1;
            name = ['test', num2str(cnt,'%03d'),num2str(cnt2,'%03d'),'.png'];
            imwrite(J,name)
        end
    end
end

toc